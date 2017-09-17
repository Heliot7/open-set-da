#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <math.h>
#include <glob.h>
#include "MetaRecognition.h"
#include "svm.h"

struct svm_node *x;
int max_nr_attr = 64;

struct svm_model* model;
struct svm_model* model_one_wsvm;
int predict_probability=0;
double min_threshold = 0, max_threshold = 0;
bool min_set = false, max_set = false;
bool verbose=true;
int debug_level=0;

static char *line = NULL;
static int max_line_len;

//Open set stuff
bool open_set = false;
int nr_classes = 0;
double *lbl;

//score/vote output
bool output_scores = false;
bool output_total_scores = false;
bool output_votes = false;

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

static char* readline(FILE *input){
	int len;
	
	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL){
		max_line_len *= 2;
		line = (char *) realloc(line,(ulong)max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}

void exit_input_error(int line_num){
	fprintf(stderr,"Wrong input format at line %d\n", line_num);
	exit(1);
}

void predict(FILE *input, FILE *output){
	int correct = 0;
	int reccorrect = 0;
    int OS_truereg=0;
    int OS_falsereg=0;
	int falsepos=0, falseneg=0, truepos=0, trueneg=0;
	int osfalsepos=0, osfalseneg=0, ostruepos=0, ostrueneg=0;
	int total = 0;
	double error = 0;
	double sump = 0, sumt = 0, sumpp = 0, sumtt = 0, sumpt = 0;
	int svm_type, nr_class;
    svm_type = svm_get_svm_type(model);
    nr_class = svm_get_nr_class(model);

	double *prob_estimates=NULL;
	int j;

	if(predict_probability && !open_set){
		if (svm_type==NU_SVR || svm_type==EPSILON_SVR)
			printf("Prob. model for test data: target value = predicted value + z,\nz: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma=%g\n",svm_get_svr_probability(model));
		else{
			int *labels=(int *) malloc(nr_class*sizeof(int));
			svm_get_labels(model,labels);
			prob_estimates = (double *) malloc(nr_class*sizeof(double));
			fprintf(output,"labels");		
			for(j=0;j<nr_class;j++)
				fprintf(output," %d",labels[j]);
			fprintf(output,"\n");
			free(labels);
		}
	}
	max_line_len = 1024;
	line = (char *)malloc(max_line_len*sizeof(char));
	while(readline(input) != NULL){
		int i = 0;
		double target_label, predict_label = 0;
		char *idx, *val, *label, *endptr;
		int inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0

		label = strtok(line," \t\n");
		if(label == NULL) // empty line
			exit_input_error(total+1);

		target_label = strtod(label,&endptr);
                //printf("Target Label %lf\n",target_label);
		if(endptr == label || *endptr != '\0')
			exit_input_error(total+1);

		while(1){
			if(i>=max_nr_attr-1){
				max_nr_attr *= 2;
				x = (struct svm_node *) realloc(x,max_nr_attr*sizeof(struct svm_node));
			}

			idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(val == NULL)
				break;
			errno = 0;
			x[i].index = (int) strtol(idx,&endptr,10);
			if(endptr == idx || errno != 0 || *endptr != '\0' || x[i].index <= inst_max_index)
				exit_input_error(total+1);
			else
				inst_max_index = x[i].index;

			errno = 0;
			x[i].value = strtod(val,&endptr);
			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(total+1);

			++i;
		}
		x[i].index = -1;

		if (predict_probability && (svm_type==C_SVC || svm_type==NU_SVC)){
			predict_label = svm_predict_probability(model,x,prob_estimates);
			fprintf(output,"%g",predict_label);
			for(j=0;j<nr_class;j++)
				fprintf(output," %g",prob_estimates[j]);
			fprintf(output,"\n");
		}
		else if (svm_type == ONE_VS_REST_WSVM){	
			int *votes = NULL;
			double **scores = Malloc(double *, nr_class+1);
                        votes = Malloc(int,nr_class+1);
			for(int v=0; v<nr_class; v++){
				scores[v] = Malloc(double, nr_class);
                                memset(scores[v],0,nr_class*sizeof(double));
			}
			predict_label = svm_predict_extended_plus_one_wsvm(model,model_one_wsvm,x, scores, votes);
			double max_prob=scores[0][0];//int max_prob_index=0;
			for(int jj=0; jj< model->openset_dim; jj++){
				if(scores[jj][0] > max_prob){
					max_prob = scores[jj][0];
				}
			}
			bool known_class=false;
			for(int jj=0; jj< model->openset_dim; jj++){
				if(target_label == model->label[jj])
					known_class=true;
			}
			if(known_class){
				if( (target_label == predict_label) && (max_prob > model->param.openset_min_probability) )
                    truepos++;
				else
                    falseneg++;
			}
			else{
				if(max_prob < model->param.openset_min_probability)
                    trueneg++;
				else
                    falsepos++;
			}
            fprintf(output,"%g: %g\n",predict_label,max_prob);
			//cleanup scores and votes
			for(int v=0; v<model->nr_class; v++)
                          if(scores[v] != NULL)
                            free(scores[v]);

			if(votes != NULL)
                          free(votes);             
		}
        else if(svm_type == ONE_WSVM || svm_type == PI_SVM){
            int *votes = NULL;
			double **scores = Malloc(double *, nr_class+1);
            votes = Malloc(int,nr_class+1);
			for(int v=0; v<nr_class; v++){
				scores[v] = Malloc(double, nr_class);
                memset(scores[v],0,nr_class*sizeof(double));
			}
			predict_label = svm_predict_extended(model,x, scores, votes);
            double max_prob=scores[0][0];
            for(int jj=0; jj< model->openset_dim; jj++){
                if(scores[jj][0] > max_prob){
                    max_prob = scores[jj][0];
                }
            }
            bool known_class=false;
            for(int jj=0; jj< model->openset_dim; jj++){
                if(target_label == model->label[jj])
                    known_class=true;
            }
            if(known_class){
                if( (target_label == predict_label) && (max_prob > model->param.openset_min_probability) )
                    truepos++;
                else
                    falseneg++;
            }
            else{
                if(max_prob < model->param.openset_min_probability)
                    trueneg++;
                else
                    falsepos++;
            }
            fprintf(output,"%g",predict_label);
            //cleanup scores and votes
            for(int v=0; v<model->nr_class; v++)
                if(scores[v] != NULL)
                    free(scores[v]);
            if(votes != NULL)
                free(votes);
        }
		else  if (svm_type == OPENSET_PAIR){
			int *votes = NULL;
			double **scores = Malloc(double *, nr_class+1);
			for(int v=0; v<nr_class; v++){
				scores[v] = Malloc(double, nr_class);
				for(int z=0; z<nr_class; z++)
					scores[v][z] = 0;
			}
			predict_label = svm_predict_extended(model,x, scores, votes);
            
            fprintf(output,"%g",predict_label);
            if(predict_label== target_label) {
                if(! (model->param.neg_labels==false && target_label>=0)){
                    reccorrect++;
                }
            }
            
            int labfound=0;
            for(int v=0; v<nr_class; v++)
                if(target_label == model->label[v]) labfound=1;
                        
            if(predict_label == model->param.rejectedID){
                if(labfound)  OS_falsereg++;
                else OS_truereg++;
            }
             
			if(output_scores || output_votes || output_total_scores){
                          if(predict_label== target_label) fprintf(output," (== %g)", target_label);
                          else fprintf(output," (!= %g)",target_label);
				if(output_votes){
					for(int v=0; v<nr_class; v++)
						fprintf(output," %d", votes[v]);
				}
				if(output_scores){
					for(int v=0; v<nr_class; v++)
						for(int z=0; z<nr_class; z++)
							if(v != z)
								fprintf(output," %d-%d:%g", v+1, z+1, scores[v][z]);
				}
				if(output_total_scores){
					double *total_scores = Malloc(double, nr_class);
					for(int v=0; v<nr_class; v++)
						total_scores[v] = 0;
					for(int v=0; v<nr_class; v++)
						for(int z=0; z<nr_class; z++)
							total_scores[v] += scores[v][z];
					for(int v=0; v<nr_class; v++)
						fprintf(output," %g", total_scores[v]);
					free(total_scores);				
				}
				fprintf(output,"\n");
			}
			else{
                          fprintf(output,"\n");
			}

            // get openset estimates for regular stuff
                        
            for(int v=0; v<nr_class; v++){
                for(int j=0; j<nr_class; j++){
                //                          fprintf(stderr,"Try for %d (lab %d) with  score %g and ",v,model->label[v],scores[v][0]);
                //                          if(model->param.neg_labels==false && model->label[v]<0 ) continue;
                //                          fprintf(stderr,"%g (!= %g) \n",predict_label, target_label);
                if(scores[v][j] <0 && model->label[v] != target_label ) ++ostrueneg;
                if(scores[v][j] >=0 && model->label[v] != target_label ) ++osfalsepos;
                if(scores[v][j] >=0 && model->label[v] == target_label) ++ostruepos;
                if(scores[v][j] < 0 && model->label[v] == target_label) ++osfalseneg;
                }
            }
                      
			if(predict_label == target_label){
				++correct; 
				if(predict_label > 0) ++truepos;
				else 
					++falseneg;
			} else {
				if(predict_label > 0) ++falsepos; 
				else 
					++trueneg;
                                } 
			//cleanup scores and votes
			for(int v=0; v<model->nr_class; v++)
				if(scores[v] != NULL)
					free(scores[v]);
			if(scores != NULL)
				free(scores);
			if(votes != NULL)
				free(votes);

		}
        else  if (!open_set){
			int *votes = NULL;
			double **scores = Malloc(double *, nr_class+1);
			for(int v=0; v<nr_class; v++){
				scores[v] = Malloc(double, nr_class);
				for(int z=0; z<nr_class; z++)
					scores[v][z] = 0;
			}
			predict_label = svm_predict_extended(model,x, scores, votes);
            
            fprintf(output,"%g",predict_label);

            
			if(predict_label == target_label)
				++correct;

			//cleanup scores and votes
			for(int v=0; v<model->nr_class; v++)
				if(scores[v] != NULL)
					free(scores[v]);
			if(scores != NULL)
				free(scores);
			if(votes != NULL)
				free(votes);
            
		} 
		else{  //open set
			int *votes = NULL;
			double **scores = Malloc(double *, nr_class+1);
            votes = Malloc(int,nr_class+1);
			for(int v=0; v<nr_class; v++){
				scores[v] = Malloc(double, nr_class);
                                memset(scores[v],0,nr_class*sizeof(double));
			}
			predict_label = svm_predict_extended(model,x, scores, votes);
			fprintf(output,"%g",predict_label);

            if(predict_label== target_label ) {
                if(! (model->param.neg_labels==false && target_label>=0)){
                    reccorrect++;
                }
            }

            int labfound=0;
            for(int v=0; v<nr_class; v++)
                if(target_label == model->label[v]) labfound=1;
                    if(predict_label == model->param.rejectedID){
                        if(labfound)  OS_falsereg++;
                        else OS_truereg++;
                    }
            for(int v=0; v<nr_class; v++){
                //                          fprintf(stderr,"Try for %d (lab %d) with ",v,model->label[v]);
                if(model->param.neg_labels==false && model->label[v]<0 ) continue;
                //                          fprintf(stderr,"%g (!= %g)\n",predict_label, target_label);
                if(scores[v][0] <0 && model->label[v] != target_label ) ostrueneg += nr_class;
                if(scores[v][0] >=0 && model->label[v] != target_label ) osfalsepos+= nr_class;
                if(scores[v][0] >=0 && model->label[v] == target_label) ostruepos+= nr_class;
                if(scores[v][0] < 0 && model->label[v] == target_label) osfalseneg+= nr_class;
            }
	
            if(output_scores || output_votes || output_total_scores){
                if(predict_label== target_label) fprintf(output," (== %g)", target_label);
                else fprintf(output," (!= %g)",target_label);
                if(output_votes){
                    for(int v=0; v<nr_class; v++)
                        fprintf(output," %d:%d", model->label[v],votes[v]);
                }
                if(output_scores){
                    for(int v=0; v<nr_class; v++){
                        fprintf(output," %d:%g", model->label[v], scores[v][0]);
                    }
                }
                if(output_total_scores){
                    for(int v=0; v<nr_class; v++)
                        fprintf(output,"  %d:%g", model->label[v], scores[v][0]);
                }
                fprintf(output,"\n");
            }
            else{
                fprintf(output,"\n");
            }

            if(model->nr_class <= 2){
                    predict_label = (predict_label>=0)?1:-1;
            }
                        
            if(predict_label == target_label){
                ++correct;
            if(predict_label > 0) ++truepos;
            else ++trueneg;
            }
            else {
                if(predict_label > 0) ++falsepos;
                else ++falseneg;
            }
        

			//cleanup scores and votes
			for(int v=0; v<model->nr_class; v++)
                          if(scores[v] != NULL)
                            free(scores[v]);
			if(scores != NULL)
                          free(scores);
			if(votes != NULL)
                          free(votes);
		}
		error += (predict_label-target_label)*(predict_label-target_label);
		sump += predict_label;
		sumt += target_label;
		sumpp += predict_label*predict_label;
		sumtt += target_label*target_label;
		sumpt += predict_label*target_label;
		++total;
	}
	
	if (svm_type==NU_SVR || svm_type==EPSILON_SVR ){
		printf("Mean squared error = %g (regression)\n",error/total);
		printf("Squared correlation coefficient = %g (regression)\n",
		       ((total*sumpt-sump*sumt)*(total*sumpt-sump*sumt))/
		       ((total*sumpp-sump*sump)*(total*sumtt-sumt*sumt))
		       );
	}
    else if(!open_set){
        printf("Accuracy = %g%% (%d/%d) (classification)\n",
               (double)correct/total*100,correct,total);
    }
	else{
        //open-set
		if(svm_type==ONE_VS_REST_WSVM || svm_type == ONE_WSVM || PI_SVM){
            double rec_acc = (double)((double)(truepos+trueneg))/((double)(truepos+trueneg+falsepos+falseneg));
            printf("Recognition Accuracy = %g%%\n",(rec_acc*100));
            double precision=0;
            if ( (truepos+falsepos) > 0)
                precision = ((double) (truepos)/(truepos+falsepos));
            double recall = 0;
            if((truepos + falseneg) > 0)
                recall = ((double) truepos)/(truepos + falseneg);
            double fmeasure = 0;
            if( (precision + recall > 0))
                fmeasure = 2* precision*recall/(precision + recall);
            printf("  Precision=%lf,   Recall=%lf   Fmeasure=%lf\n",precision, recall, fmeasure);
            printf("  Total tests=%d, True pos %d True Neg %d, False Pos %d, False neg %d\n",
                               truepos+ trueneg+ falsepos+ falseneg, truepos, trueneg, falsepos, falseneg);
		}		
		else
		{
		
            if(nr_classes > 1)
                printf("Classification (Multi-class Recognition)  Rate = %g%% (%d/%d)\n",
                       (double)correct/total*100,correct,total);
            else
                printf("Classification Accuracy = %g%% (%d/%d)\n",
								(double)correct/total*100,correct,total);

            if(open_set || verbose || (truepos+falsepos >0)){
                if ( (truepos+falsepos) > 0){
                    double precision = ((double) (truepos)/(truepos+falsepos));
                    double recall = 0;
                if((truepos + falseneg) > 0) recall = ((double) truepos)/(truepos + falseneg);
                double fmeasure = 0;
                if( (precision + recall > 0)) fmeasure = 2* precision*recall/(precision + recall);
                printf("  Precision=%lf,   Recall=%lf   Fmeasure=%lf\n",precision, recall, fmeasure);
                if(verbose)
                    printf("   Total tests=%d, True pos %d True Neg %d, False Pos %d, False neg %d\n",
						   truepos+ trueneg+ falsepos+ falseneg, truepos, trueneg, falsepos, falseneg);
                }
                else if(((truepos+falsepos)==0)){
                    printf("  Precision=0,   Recall=0   Fmeasure=0\n");
                }
                if ( (ostruepos+osfalsepos) > 0){
                    double precision = ((double) (ostruepos)/(ostruepos+osfalsepos));
                    double recall = 0;
                    if((ostruepos + osfalseneg) > 0) recall = ((double) ostruepos)/(ostruepos + osfalseneg);
                    double fmeasure = 0;
                    if( (precision + recall > 0)) fmeasure = 2* precision*recall/(precision + recall);
                    if(verbose)
                        printf("   Total Pairwise tests=%d, True pos %d True Neg %d, False Pos %d, False neg %d\n",
						   ostruepos+ ostrueneg+ osfalsepos+ osfalseneg, ostruepos, ostrueneg, osfalsepos, osfalseneg);
                    printf("  Pairwise Precision=%lf,   Recall=%lf   Fmeasure=%lf\n",precision, recall, fmeasure);
			
                }
                else if(((ostruepos+osfalsepos)==0))
                    printf("  Pariwise Precision=0,   Recall=0   Fmeasure=0\n");
                
                if(reccorrect >0) {
                    printf("Multiclass Recognition Rate   =   %g%% (%d/%d)\n",
					   (double)100*reccorrect/(total),reccorrect,total );
                    printf("Multiclass Recognition Recall  =   %g%% (%d/%d)\n\n",
					   (double)ostruepos/(ostruepos+osfalseneg)* 100,ostruepos,ostruepos+osfalseneg );
                }
                if(OS_truereg+ OS_falsereg>0)
                    printf("Unknown classes true rejections %d, False rejections %d\n\n",
                                 OS_truereg, OS_falsereg );
            }
		}
	}

	if(predict_probability)
		free(prob_estimates);
}

void exit_with_help(){
	printf(
	"Usage: svm-predict [options] test_file model_file output_file\n"
	"options:\n"
	"  -b probability_estimates: whether to predict probability estimates, 0 or 1 (default 0); for one-class SVM only 0 is supported\n"
    "  -o: this is an open set problem. this will look for model files with names of the form <model_file>.<class>\n"
    "  -V  for more verbose output\n"
	"  -s output scores in bin format(1-2, 1-3, 1-4, 2-3) to outputfile(cannot be combined with -v or -t) \n"
	"  -t output totaled scores 1-2+1-3+1-4=1 ect to outputfile(cannot be combined with -s or -v) \n"
	"  -v output votes to outputfile(cannot be combined with -s or -t) \n"
    "  -P threshold probability value to reject sample as unknowns for WSVM(default 0.0) \n"
    "  -C threshold probability value to reject sample as unknowns for CAP model in WSVM(default 0.0) \n"
	);
	exit(1);
}

int main(int argc, char **argv)
{
	FILE *input, *output;
	int i;
    double openset_min_probability=0.0;
	double openset_min_probability_one_wsvm=0.00;
	char model_file_name_one_wsvm[1024];

	// parse options
	for(i=1;i<argc;i++){
		if(argv[i][0] != '-') break;
		switch(argv[i][1]){
                case 'b':
                  predict_probability = atoi(argv[++i]);
                  break;
                case 'P':
                  openset_min_probability = atof(argv[++i]);
                  break;
                case 'C':
                  openset_min_probability_one_wsvm = atof(argv[++i]);
                  break;
                case 'o':
                  open_set = true;
                  break;
                case 'V':
                  verbose = true;
                  break;
				case 's':
					output_scores = true;
					break;
				case 'a':
					output_scores = true;
					output_votes = true;
					output_total_scores = true;
					break;
				case 't':
					output_total_scores = true;
					break;
				case 'v':
					output_votes = true;
					break;
                default:
                  fprintf(stderr,"Unknown option: -%c\n", argv[i][1]);
                  exit_with_help();
		}
	}
	if(i>argc-2)
		exit_with_help();
	input = fopen(argv[i],"r");
	if(input == NULL){
		fprintf(stderr,"can't open input file %s\n",argv[i]);
		exit(1);
	}

	output = fopen(argv[i+2],"w");
	if(output == NULL){
		fprintf(stderr,"can't open output file %s\n",argv[i+2]);
		exit(1);
	}
    if((model=svm_load_model(argv[i+1]))==0){
        fprintf(stderr,"can't open model file %s\n",argv[i+1]);
        exit(1);
    }

    if (model->param.svm_type == ONE_VS_REST_WSVM){
        strcpy(model_file_name_one_wsvm,argv[i+1]);
        strcat(model_file_name_one_wsvm,"_one_wsvm");
        if((model_one_wsvm=svm_load_model(model_file_name_one_wsvm))==0){
            fprintf(stderr,"can't open model file %s\n",model_file_name_one_wsvm);
            exit(1);
        }
        model_one_wsvm->param.openset_min_probability = openset_min_probability_one_wsvm;

    }
    model->param.openset_min_probability = openset_min_probability;
    if(model && (model->param.svm_type == OPENSET_OC || model->param.svm_type == OPENSET_BIN || model->param.svm_type == OPENSET_PAIR ||model->param.svm_type == ONE_VS_REST_WSVM ||model->param.svm_type == ONE_WSVM || model->param.svm_type == PI_SVM))
		open_set=true;

	x = (struct svm_node *) malloc(max_nr_attr*sizeof(struct svm_node));
	if(predict_probability && !open_set){
		if(svm_check_probability_model(model)==0){
			fprintf(stderr,"Model does not support probabiliy estimates\n");
			exit(1);
		}
		predict(input,output);
	}
	else
		predict(input,output);
    
	if(model->param.svm_type == ONE_VS_REST_WSVM )
        svm_free_and_destroy_model(&model_one_wsvm);
    svm_free_and_destroy_model(&model);


	free(x);
	free(line);
	fclose(input);
	fclose(output);
	return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "svm.h"

#include "mex.h"
#include "svm_model_matlab.h"

#if MX_API_VER < 0x07030000
typedef int mwIndex;
#endif

#define CMD_LEN 2048
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

int prob_estimate_flag = 0;
double openset_min_probability = 0.0;
double openset_min_probability_one_wsvm = 0.00;
struct svm_model *model;
struct svm_model* model_one_wsvm;

bool open_set = false;
int nr_classes = 0;
double min_threshold = 0, max_threshold = 0;
bool min_set = false, max_set = false;
bool verbose = true;
int debug_level = 0;
bool output_scores = false;
bool output_total_scores = false;
bool output_votes = false;
	
void read_sparse_instance(const mxArray *prhs, int index, struct svm_node *x)
{
	int i, j, low, high;
	mwIndex *ir, *jc;
	double *samples;

	ir = mxGetIr(prhs);
	jc = mxGetJc(prhs);
	samples = mxGetPr(prhs);

	// each column is one instance
	j = 0;
	low = (int)jc[index], high = (int)jc[index+1];
	for(i=low;i<high;i++)
	{
		x[j].index = (int)ir[i] + 1;
		x[j].value = samples[i];
		j++;
	}
	x[j].index = -1;
}

static void fake_answer(mxArray *plhs[])
{
	plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
	plhs[1] = mxCreateDoubleMatrix(0, 0, mxREAL);
	plhs[2] = mxCreateDoubleMatrix(0, 0, mxREAL);
}

void predict(mxArray *plhs[], const mxArray *prhs[], const int predict_probability)
{
	int label_vector_row_num, label_vector_col_num;
	int feature_number, testing_instance_number;
	int instance_index;
	double *ptr_instance, *ptr_label, *ptr_predict_label; 
	double *ptr_prob_estimates, *ptr_dec_values, *ptr;
	struct svm_node *x;
	mxArray *pplhs[1]; // transposed instance sparse matrix

	int correct = 0;
	int reccorrect = 0;
    int OS_truereg=0;
    int OS_falsereg=0;
	int falsepos=0, falseneg=0, truepos=0, trueneg=0;
	int osfalsepos=0, osfalseneg=0, ostruepos=0, ostrueneg=0;
	int total = 0;
	double error = 0;
	double sump = 0, sumt = 0, sumpp = 0, sumtt = 0, sumpt = 0;

	int svm_type=svm_get_svm_type(model);
	int nr_class=svm_get_nr_class(model);
	double *prob_estimates=NULL;

	// prhs[1] = testing instance matrix
	feature_number = (int)mxGetN(prhs[1]);
	testing_instance_number = (int)mxGetM(prhs[1]);
	label_vector_row_num = (int)mxGetM(prhs[0]);
	label_vector_col_num = (int)mxGetN(prhs[0]);

	if(label_vector_row_num!=testing_instance_number)
	{
		mexPrintf("Length of label vector does not match # of instances.\n");
		fake_answer(plhs);
		return;
	}
	if(label_vector_col_num!=1)
	{
		mexPrintf("label (1st argument) should be a vector (# of column is 1).\n");
		fake_answer(plhs);
		return;
	}

	ptr_instance = mxGetPr(prhs[1]);
	ptr_label    = mxGetPr(prhs[0]);

	// transpose instance matrix
	if(mxIsSparse(prhs[1]))
	{
		if(model->param.kernel_type == PRECOMPUTED)
		{
			// precomputed kernel requires dense matrix, so we make one
			mxArray *rhs[1], *lhs[1];
			rhs[0] = mxDuplicateArray(prhs[1]);
			if(mexCallMATLAB(1, lhs, 1, rhs, "full"))
			{
				mexPrintf("Error: cannot full testing instance matrix\n");
				fake_answer(plhs);
				return;
			}
			ptr_instance = mxGetPr(lhs[0]);
			mxDestroyArray(rhs[0]);
		}
		else
		{
			mxArray *pprhs[1];
			pprhs[0] = mxDuplicateArray(prhs[1]);
			if(mexCallMATLAB(1, pplhs, 1, pprhs, "transpose"))
			{
				mexPrintf("Error: cannot transpose testing instance matrix\n");
				fake_answer(plhs);
				return;
			}
		}
	}

	if(predict_probability && !open_set)
	{
		if(svm_type==NU_SVR || svm_type==EPSILON_SVR)
			mexPrintf("Prob. model for test data: target value = predicted value + z,\nz: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma=%g\n",svm_get_svr_probability(model));
		else
			prob_estimates = (double *) malloc(nr_class*sizeof(double));
	}

	plhs[0] = mxCreateDoubleMatrix(testing_instance_number, 1, mxREAL);
	if(predict_probability)
	{
		// prob estimates are in plhs[2]
		if(svm_type==C_SVC || svm_type==NU_SVC)
			plhs[2] = mxCreateDoubleMatrix(testing_instance_number, nr_class, mxREAL);
		else
			plhs[2] = mxCreateDoubleMatrix(0, 0, mxREAL);
	}
	else
	{
		// decision values are in plhs[2]
		if(svm_type == ONE_CLASS ||
		   svm_type == EPSILON_SVR ||
		   svm_type == NU_SVR)
			plhs[2] = mxCreateDoubleMatrix(testing_instance_number, 1, mxREAL);
		else
			plhs[2] = mxCreateDoubleMatrix(testing_instance_number, nr_class*(nr_class-1)/2, mxREAL);
	}

	ptr_predict_label = mxGetPr(plhs[0]);
	ptr_prob_estimates = mxGetPr(plhs[2]);
	ptr_dec_values = mxGetPr(plhs[2]);
	x = (struct svm_node*)malloc((feature_number+1)*sizeof(struct svm_node) );
	for(instance_index = 0; instance_index < testing_instance_number; instance_index++)
	{
		int i;
		double target_label, predict_label;

		target_label = ptr_label[instance_index];

		if(mxIsSparse(prhs[1]) && model->param.kernel_type != PRECOMPUTED) // prhs[1]^T is still sparse
			read_sparse_instance(pplhs[0], instance_index, x);
		else
		{
			for(i=0;i<feature_number;i++)
			{
				x[i].index = i+1;
				x[i].value = ptr_instance[testing_instance_number*i+instance_index];
			}
			x[feature_number].index = -1;
		}

		if(predict_probability && (svm_type==C_SVC || svm_type==NU_SVC))
		{
			predict_label = svm_predict_probability(model, x, prob_estimates);
			ptr_predict_label[instance_index] = predict_label;
			for(i=0;i<nr_class;i++)
				ptr_prob_estimates[instance_index + i * testing_instance_number] = prob_estimates[i];
		}
		//else if(predict_probability && !(svm_type==C_SVC || svm_type==NU_SVC))
		//{
		//	predict_label = svm_predict(model,x);
		//	ptr_predict_label[instance_index] = predict_label;
		//}
		else if (svm_type == ONE_VS_REST_WSVM)
		{
			int *votes = NULL;
			double **scores = Malloc(double *, nr_class+1);
			votes = Malloc(int,nr_class+1);
			for(int v = 0; v < nr_class; v++)
			{
				scores[v] = Malloc(double, nr_class);
				memset(scores[v],0,nr_class*sizeof(double));
			}
			predict_label = svm_predict_extended_plus_one_wsvm(model, model_one_wsvm, x, scores, votes);
			if(instance_index == 0)
			{
				//for (int idx = 0; idx < feature_number; ++idx)
					//mexPrintf("%.2f ", x[idx].value);
				// mexPrintf("\n");
				// mexPrintf("predict_label: %f\n", predict_label);
			}
			double max_prob = scores[0][0];//int max_prob_index=0;
			for(int jj = 0; jj < model->openset_dim; jj++){
				if(scores[jj][0] > max_prob)
				{					
					max_prob = scores[jj][0];
				}
				if(instance_index == 0)
				{
					// mexPrintf("%.18f ", scores[jj][0]);
					// mexPrintf("%.2f ", votes[jj]);
				}
			}
			// if(instance_index == 0)
				//mexPrintf("\n");
			bool known_class = false;
			for(int jj=0; jj< model->openset_dim; jj++)
			{
				if(target_label == model->label[jj])
					known_class = true;
			}
			if(known_class)
			{
				if( (target_label == predict_label) && (max_prob > model->param.openset_min_probability) )
                    truepos++;
				else
                    falseneg++;
			}
			else
			{
				if(max_prob < model->param.openset_min_probability)
                    trueneg++;
				else
                    falsepos++;
			}
			
			// PAU: Change to matlab output
            // fprintf(output,"%g: %g\n",predict_label,max_prob);
			ptr_predict_label[instance_index] = predict_label;
			for(i = 0; i < nr_class; i++) // Copy max in all label spaces?
				ptr_prob_estimates[instance_index + i * testing_instance_number] = 0;
			ptr_prob_estimates[instance_index + ((int)predict_label-1) * testing_instance_number] = max_prob;
			
			//cleanup scores and votes
			for(int v=0; v<model->nr_class; v++)
				if(scores[v] != NULL)
				free(scores[v]);

			if(votes != NULL)
				free(votes);            
		}
		else if(svm_type == ONE_WSVM || svm_type == PI_SVM)
		{
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
			// PAU: Change to matlab output
			// fprintf(output,"%g",predict_label);
			ptr_predict_label[instance_index] = predict_label;
            //cleanup scores and votes
            for(int v=0; v<model->nr_class; v++)
                if(scores[v] != NULL)
                    free(scores[v]);
            if(votes != NULL)
                free(votes);
        }
		else if(svm_type == OPENSET_PAIR)
		{
			int *votes = NULL;
			double **scores = Malloc(double *, nr_class+1);
			for(int v=0; v<nr_class; v++){
				scores[v] = Malloc(double, nr_class);
				for(int z=0; z<nr_class; z++)
					scores[v][z] = 0;
			}
			predict_label = svm_predict_extended(model,x, scores, votes);
            
			// PAU: Change to matlab output
			// fprintf(output,"%g",predict_label);
			ptr_predict_label[instance_index] = predict_label;
            if(predict_label == target_label)
			{
                if(! (model->param.neg_labels==false && target_label>=0))
				{
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
			
			// PAU: Change to matlab output
			if(output_scores || output_votes || output_total_scores)
			{
				if(predict_label == target_label)
					mexPrintf(" (== %g)\n", target_label);
				else
					mexPrintf(" (!= %g)\n",target_label);
				if(output_votes)
				{
					for(int v=0; v<nr_class; v++)
						mexPrintf(" %d\n", votes[v]);
				}
				if(output_scores)
				{
					for(int v=0; v<nr_class; v++)
						for(int z=0; z<nr_class; z++)
							if(v != z)
								mexPrintf(" %d-%d:%g\n", v+1, z+1, scores[v][z]);
				}
				if(output_total_scores)
				{
					double *total_scores = Malloc(double, nr_class);
					for(int v=0; v<nr_class; v++)
						total_scores[v] = 0;
					for(int v=0; v<nr_class; v++)
						for(int z=0; z<nr_class; z++)
							total_scores[v] += scores[v][z];
					for(int v=0; v<nr_class; v++)
						mexPrintf(" %g\n", total_scores[v]);
					free(total_scores);
				}
				mexPrintf("\n");
			}
			else
			{
				mexPrintf("\n");
			}
            // get openset estimates for regular stuff 
            for(int v=0; v<nr_class; v++)
			{
                for(int j=0; j<nr_class; j++)
				{
					//                          fprintf(stderr,"Try for %d (lab %d) with  score %g and ",v,model->label[v],scores[v][0]);
					//                          if(model->param.neg_labels==false && model->label[v]<0 ) continue;
					//                          fprintf(stderr,"%g (!= %g) \n",predict_label, target_label);
					if(scores[v][j] <0 && model->label[v] != target_label ) ++ostrueneg;
					if(scores[v][j] >=0 && model->label[v] != target_label ) ++osfalsepos;
					if(scores[v][j] >=0 && model->label[v] == target_label) ++ostruepos;
					if(scores[v][j] < 0 && model->label[v] == target_label) ++osfalseneg;
                }
            }
                      
			if(predict_label == target_label)
			{
				++correct; 
				if(predict_label > 0) ++truepos;
				else 
					++falseneg;
			} else
			{
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
		else
		{
			if(svm_type == ONE_CLASS ||
			   svm_type == EPSILON_SVR ||
			   svm_type == NU_SVR)
			{
				double res;
				predict_label = svm_predict_values(model, x, &res);
				ptr_dec_values[instance_index] = res;
			}
			else
			{
				double *dec_values = (double *) malloc(sizeof(double) * nr_class*(nr_class-1)/2);
				predict_label = svm_predict_values(model, x, dec_values);
				for(i=0;i<(nr_class*(nr_class-1))/2;i++)
					ptr_dec_values[instance_index + i * testing_instance_number] = dec_values[i];
				free(dec_values);
			}
			ptr_predict_label[instance_index] = predict_label;
		}

		if(predict_label == target_label)
			++correct;
		error += (predict_label-target_label)*(predict_label-target_label);
		sump += predict_label;
		sumt += target_label;
		sumpp += predict_label*predict_label;
		sumtt += target_label*target_label;
		sumpt += predict_label*target_label;
		++total;
	}
	
	if(svm_type==NU_SVR || svm_type==EPSILON_SVR)
	{
		mexPrintf("Mean squared error = %g (regression)\n",error/total);
		mexPrintf("Squared correlation coefficient = %g (regression)\n",
			((total*sumpt-sump*sumt)*(total*sumpt-sump*sumt))/
			((total*sumpp-sump*sump)*(total*sumtt-sumt*sumt))
			);
	}
	else if(!open_set)
	{
		mexPrintf("Accuracy = %g%% (%d/%d) (classification)\n",
			(double)correct/total*100,correct,total);
	}
	else // Open-set
	{
		/*
		if(svm_type==ONE_VS_REST_WSVM || svm_type == ONE_WSVM || PI_SVM){
            double rec_acc = (double)((double)(truepos+trueneg))/((double)(truepos+trueneg+falsepos+falseneg));
            mexPrintf("Recognition Accuracy = %g%%\n",(rec_acc*100));
            double precision=0;
            if ( (truepos+falsepos) > 0)
                precision = ((double) (truepos)/(truepos+falsepos));
            double recall = 0;
            if((truepos + falseneg) > 0)
                recall = ((double) truepos)/(truepos + falseneg);
            double fmeasure = 0;
            if( (precision + recall > 0))
                fmeasure = 2* precision*recall/(precision + recall);
            mexPrintf("  Precision=%lf,   Recall=%lf   Fmeasure=%lf\n",precision, recall, fmeasure);
            mexPrintf("  Total tests=%d, True pos %d True Neg %d, False Pos %d, False neg %d\n",
                               truepos+ trueneg+ falsepos+ falseneg, truepos, trueneg, falsepos, falseneg);
		}		
		else
		{
            if(nr_classes > 1)
                mexPrintf("Classification (Multi-class Recognition)  Rate = %g%% (%d/%d)\n",
                       (double)correct/total*100,correct,total);
            else
                mexPrintf("Classification Accuracy = %g%% (%d/%d)\n",
								(double)correct/total*100,correct,total);

            if(open_set || verbose || (truepos+falsepos >0)){
                if ( (truepos+falsepos) > 0){
                    double precision = ((double) (truepos)/(truepos+falsepos));
                    double recall = 0;
                if((truepos + falseneg) > 0) recall = ((double) truepos)/(truepos + falseneg);
                double fmeasure = 0;
                if( (precision + recall > 0)) fmeasure = 2* precision*recall/(precision + recall);
                mexPrintf("  Precision=%lf,   Recall=%lf   Fmeasure=%lf\n",precision, recall, fmeasure);
                if(verbose)
                    mexPrintf("   Total tests=%d, True pos %d True Neg %d, False Pos %d, False neg %d\n",
						   truepos+ trueneg+ falsepos+ falseneg, truepos, trueneg, falsepos, falseneg);
                }
                else if(((truepos+falsepos)==0)){
                    mexPrintf("  Precision=0,   Recall=0   Fmeasure=0\n");
                }
                if ( (ostruepos+osfalsepos) > 0){
                    double precision = ((double) (ostruepos)/(ostruepos+osfalsepos));
                    double recall = 0;
                    if((ostruepos + osfalseneg) > 0) recall = ((double) ostruepos)/(ostruepos + osfalseneg);
                    double fmeasure = 0;
                    if( (precision + recall > 0)) fmeasure = 2* precision*recall/(precision + recall);
                    if(verbose)
                        mexPrintf("   Total Pairwise tests=%d, True pos %d True Neg %d, False Pos %d, False neg %d\n",
						   ostruepos+ ostrueneg+ osfalsepos+ osfalseneg, ostruepos, ostrueneg, osfalsepos, osfalseneg);
                    mexPrintf("  Pairwise Precision=%lf,   Recall=%lf   Fmeasure=%lf\n",precision, recall, fmeasure);
			
                }
                else if(((ostruepos+osfalsepos)==0))
                    mexPrintf("  Pariwise Precision=0,   Recall=0   Fmeasure=0\n");
                
                if(reccorrect >0) {
                    mexPrintf("Multiclass Recognition Rate   =   %g%% (%d/%d)\n",
					   (double)100*reccorrect/(total),reccorrect,total );
                    mexPrintf("Multiclass Recognition Recall  =   %g%% (%d/%d)\n\n",
					   (double)ostruepos/(ostruepos+osfalseneg)* 100,ostruepos,ostruepos+osfalseneg );
                }
                if(OS_truereg+ OS_falsereg>0)
                    mexPrintf("Unknown classes true rejections %d, False rejections %d\n\n", OS_truereg, OS_falsereg );
            }
		}
		*/
	}
	// return accuracy, mean squared error, squared correlation coefficient
	plhs[1] = mxCreateDoubleMatrix(3, 1, mxREAL);
	ptr = mxGetPr(plhs[1]);
	ptr[0] = (double)correct/total*100;
	ptr[1] = error/total;
	ptr[2] = ((total*sumpt-sump*sumt)*(total*sumpt-sump*sumt))/
				((total*sumpp-sump*sump)*(total*sumtt-sumt*sumt));

	free(x);
	if(prob_estimates != NULL)
		free(prob_estimates);
}

void exit_with_help()
{
	mexPrintf(
		"Usage: [predicted_label, accuracy, decision_values/prob_estimates] = svmpredict(testing_label_vector, testing_instance_matrix, model, 'libsvm_options')\n"
		"Parameters:\n"
		"  model: SVM model structure from svmtrain.\n"
		"  libsvm_options:\n"
		"    -b probability_estimates: whether to predict probability estimates, 0 or 1 (default 0); one-class SVM not supported yet\n"
		"Returns:\n"
		"  predicted_label: SVM prediction output vector.\n"
		"  accuracy: a vector with accuracy, mean squared error, squared correlation coefficient.\n"
		"  prob_estimates: If selected, probability estimate vector.\n"
	);
}

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{	
	if(nrhs > 5 || nrhs < 4)
	{
		exit_with_help();
		fake_answer(plhs);
		return;
	}
	
	if(!mxIsDouble(prhs[0]) || !mxIsDouble(prhs[1])) {
		mexPrintf("Error: label vector and instance matrix must be double\n");
		fake_answer(plhs);
		return;
	}

	if(mxIsStruct(prhs[2]))
	{
		const char *error_msg;

		// parse options
		if(nrhs==5)
		{
			int i, argc = 1;
			char cmd[CMD_LEN], *argv[CMD_LEN/2];

			// put options in argv[]
			mxGetString(prhs[4], cmd,  mxGetN(prhs[4]) + 1);
			if((argv[argc] = strtok(cmd, " ")) != NULL)
				while((argv[++argc] = strtok(NULL, " ")) != NULL)
					;

			for(i=1;i<argc;i++)
			{
				if(argv[i][0] != '-') break;
				if(++i>=argc)
				{
					exit_with_help();
					fake_answer(plhs);
					return;
				}
				switch(argv[i-1][1])
				{
					case 'b':
						prob_estimate_flag = atoi(argv[i]);
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
						mexPrintf("Unknown option: -%c\n", argv[i-1][1]);
						exit_with_help();
						fake_answer(plhs);
						return;
				}
			}
		}

		// mexPrintf("start copy\n");
		model = matlab_matrix_to_model(prhs[2], &error_msg);
		// mexPrintf("end  copy\n");
		if (model == NULL)
		{
			mexPrintf("Error: can't read model: %s\n", error_msg);
			fake_answer(plhs);
			return;
		}
		if (model->param.svm_type == ONE_VS_REST_WSVM)
		{
			model_one_wsvm = matlab_matrix_to_model(prhs[3], &error_msg);
			model_one_wsvm->param.openset_min_probability = openset_min_probability_one_wsvm;
			if (model_one_wsvm == NULL)
			{
				mexPrintf("Error: can't read model: %s\n", error_msg);
				fake_answer(plhs);
				return;
			}
		}
		model->param.openset_min_probability = openset_min_probability;
		if(model && (model->param.svm_type == OPENSET_OC || model->param.svm_type == OPENSET_BIN || model->param.svm_type == OPENSET_PAIR || 
		model->param.svm_type == ONE_VS_REST_WSVM || model->param.svm_type == ONE_WSVM || model->param.svm_type == PI_SVM))
			open_set = true;
	
		if(prob_estimate_flag)
		{
			if(svm_check_probability_model(model)==0)
			{
				mexPrintf("Model does not support probabiliy estimates\n");
				fake_answer(plhs);
				svm_free_and_destroy_model(&model);
				return;
			}
		}
		else
		{
			if(svm_check_probability_model(model)!=0)
				mexPrintf("Model supports probability estimates, but disabled in predicton.\n");
		}

		// mexPrintf("PREDICTION\n");
		predict(plhs, prhs, prob_estimate_flag);
		// mexPrintf("PREDICTION END\n");
		// destroy model
		svm_free_and_destroy_model(&model);
		svm_free_and_destroy_model(&model_one_wsvm);
	}
	else
	{
		mexPrintf("model file should be a struct array\n");
		fake_answer(plhs);
	}

	return;
}

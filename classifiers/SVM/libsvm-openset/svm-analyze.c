#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <math.h>
#include "svm.h"

#define MIN(x, y) (x < y ? x : y)

struct score_data
{
    double label;
    double score;
};

struct svm_node *x;
int max_nr_attr = 64;

extern struct svm_model* model;
int predict_probability=0;
double min_threshold = 0, max_threshold = 0;
bool min_set = false, max_set = false;
int num_steps = 20;

static char *line = NULL;
static int max_line_len;

extern static char* readline(FILE *input);
void exit_input_error(int line_num);

void analyze(FILE *input, FILE *output)
{
	int correct = 0;
	int total = 0, inclass = 0;
	double error = 0;
	double sump = 0, sumt = 0, sumpp = 0, sumtt = 0, sumpt = 0;

	int svm_type=svm_get_svm_type(model);
	int nr_class=svm_get_nr_class(model);
	double *prob_estimates=NULL;
	int j;
    int max_scores = 64;
    struct score_data * scores = (struct score_data *) malloc(max_scores*sizeof(struct score_data));

	max_line_len = 1024;
	line = (char *)malloc(max_line_len*sizeof(char));
	while(readline(input) != NULL)
	{
		int i = 0;
		double target_label, predict_label;
		char *idx, *val, *label, *endptr;
		int inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0

        //Make sure we don't go over the bounds of our score array
        if (total >= max_scores)
        {
            max_scores *= 2;
            scores = (struct score_data *) realloc(scores, max_scores*sizeof(struct score_data));
        }

		label = strtok(line," \t\n");
		if(label == NULL) // empty line
			exit_input_error(total+1);

		target_label = strtod(label,&endptr);
		if(endptr == label || *endptr != '\0')
			exit_input_error(total+1);

        if (target_label > 0) ++inclass;

		while(1)
		{
			if(i>=max_nr_attr-1)	// need one more for index = -1
			{
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

		predict_label = svm_predict(model,x);
		//printf("%g %g\n", target_label, predict_label);

        scores[total].label = target_label;
        scores[total].score = predict_label;

		if((predict_label/fabs(predict_label)) == target_label)
			++correct;
		error += (predict_label-target_label)*(predict_label-target_label);
		sump += predict_label;
		sumt += target_label;
		sumpp += predict_label*predict_label;
		sumtt += target_label*target_label;
		sumpt += predict_label*target_label;
		++total;
	}

    //We have scores saved to file
    //It's easier to just read them back in
    double ma = 0, mi = 1; //Reasonable bounds on max/min to prevent outliers
                           //from causing a poor p/r analysis.
    double step;
    double min_error = 0, min_error_m = 0, min_error_t = 0;

    for (int i = 0; i < total; i++)
    {
        ma = scores[i].score > ma ? scores[i].score : ma;
        mi = scores[i].score < mi ? scores[i].score : mi;
    }

    if (min_set)
        mi = mi > min_threshold ? mi : min_threshold;
    if (max_set)
        ma = ma < max_threshold ? ma : max_threshold;

    //fprintf(output, "%g, %g\n", ma, mi);
    step = (ma - mi)/num_steps;
    //Compute precision and recall
    
    for (int i = 0; i <= num_steps-1; i++)
    {
        double tl = mi + i*step;
        for (int j = i+1; j <= num_steps; j++)
        {
            double tu = mi + j*step;
            int retrieved = 0, relevant = 0;
            double precision = 0, recall = 0, fmeasure = 0, error = 0;
            double false_accept = 0, false_reject = 0;

            for (int i = 0; i < total; i++)
            {
                if (scores[i].score >= tl && scores[i].score <= tu)
                {
                    retrieved++;
                    if (scores[i].label > 0) // equals 1
                        relevant++;
                    else
                        false_accept += MIN(fabs(scores[i].score - tl), fabs(tu - scores[i].score));
                        //false_accept++;
                }
                else
                {
                    if (scores[i].label > 0)
                        false_reject += MIN(fabs(scores[i].score - tl), fabs(tu - scores[i].score));
                        //false_reject++;
                }
            }

            error = fabs(false_accept - false_reject);

            if (retrieved > 0)
                precision = ((double) relevant)/retrieved;
            else
                precision = 0;

            recall = ((double) relevant)/inclass;

            fmeasure = 2*precision*recall/(precision + recall);
            if (precision > min_error)
            {
                min_error = precision;
                min_error_m = tl;
                min_error_t = tu;
            }

            //fprintf(output, "%g\t%g\t%g\t%g\n", precision, recall, tl, tu);
        }
    }

    free(scores);

    printf("min: %g, max: %g, error: %g\n", min_error_m, min_error_t, min_error);
}

/*
int main(int argc, char **argv)
{
	FILE *input, *output;
	int i;

	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		++i;
		switch(argv[i-1][1])
		{
			case 'b':
				predict_probability = atoi(argv[i]);
				break;
            case 'm':
                min_threshold = atof(argv[i]);
                min_set = true;
                break;
            case 't':
                max_threshold = atof(argv[i]);
                max_set = true;
                break;
            case 'n':
                num_steps = atoi(argv[i]);
                break;
			default:
				fprintf(stderr,"Unknown option: -%c\n", argv[i-1][1]);
				exit_with_help();
		}
	}
	if(i>=argc-2)
		exit_with_help();
	
	input = fopen(argv[i],"r");
	if(input == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",argv[i]);
		exit(1);
	}

	output = fopen(argv[i+2],"w+");
	if(output == NULL)
	{
		fprintf(stderr,"can't open output file %s\n",argv[i+2]);
		exit(1);
	}

	if((model=svm_load_model(argv[i+1]))==0)
	{
		fprintf(stderr,"can't open model file %s\n",argv[i+1]);
		exit(1);
	}

	x = (struct svm_node *) malloc(max_nr_attr*sizeof(struct svm_node));
	if(predict_probability)
	{
		if(svm_check_probability_model(model)==0)
		{
			fprintf(stderr,"Model does not support probabiliy estimates\n");
			exit(1);
		}
	}
	else
	{
		if(svm_check_probability_model(model)!=0)
			printf("Model supports probability estimates, but disabled in prediction.\n");
	}
	predict(input,output);
	svm_free_and_destroy_model(&model);
	free(x);
	free(line);
	fclose(input);
	fclose(output);
	return 0;
}
*/

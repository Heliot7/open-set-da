#include <stdlib.h>
#include <string.h>
#include "svm.h"

#include "mex.h"
#include "libMR\MetaRecognition.h"
#if MX_API_VER < 0x07030000
typedef int mwIndex;
#endif

#define NUM_OF_RETURN_FIELD 18

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

static const char *field_names[] = {
	"Parameters",
	"nr_class",
	"totalSV",
	// EXTRA OPEN_SET
	"openset_dim",
	// EXTRA OPEN_SET (END)
	"rho",
	"Label",
	// EXTRA OPEN_SET
	"Neg_labels",
	"Rejected_ID",
	"alpha",
	"omega",
	"MR_pos_one_vs_all",
	"MR_comp_one_vs_all",
	"MR_pos_one_class",
	// EXTRA OPEN_SET (END)
	"ProbA",
	"ProbB",
	"nSV",
	"sv_coef",
	"SVs"
};

void saveMR(double* ptr, int pos, int jump, MetaRecognition meta)
{	
	/*
	fprintf(outputFile, "%21.18g %21.18g  " //parmaht
                  "%21.18g %21.18g " //parmci 
                  "%21.18g %21.18g  "
                  "%d %f %d %d "  //sign, alpha, fitting size
                  "%lf %21.18g %d\n", //translate,  small_score, scores_to_drop
                  parmhat[0], parmhat[1],
                  parmci[0],parmci[1],
                  parmci[2],parmci[3],
                  sign, alpha, (int) ftype,fitting_size,
                  translate_amount, small_score, scores_to_drop);
	*/
	ptr[0*jump + pos] = meta.parmhat[0];
	ptr[1*jump + pos] = meta.parmhat[1];
	ptr[2*jump + pos] = meta.parmci[0];
	ptr[3*jump + pos] = meta.parmci[1];
	ptr[4*jump + pos] = meta.parmci[2];
	ptr[5*jump + pos] = meta.parmci[3];
	ptr[6*jump + pos] = meta.sign;
	ptr[7*jump + pos] = meta.alpha;
	ptr[8*jump + pos] = (int) meta.ftype;
	ptr[9*jump + pos] = meta.fitting_size;
	ptr[10*jump + pos] = meta.translate_amount;
	ptr[11*jump + pos] = meta.small_score;
	ptr[12*jump + pos] = meta.scores_to_drop;
}

void loadMR(double* ptr, int pos, int jump, MetaRecognition* meta)
{
	meta[pos].parmhat[0] = ptr[0*jump + pos];
	meta[pos].parmhat[1] = ptr[1*jump + pos];
	meta[pos].parmci[0] = ptr[2*jump + pos];
	meta[pos].parmci[1] = ptr[3*jump + pos];
	meta[pos].parmci[2] = ptr[4*jump + pos];
	meta[pos].parmci[3] = ptr[5*jump + pos];
	meta[pos].sign = (int)ptr[6*jump + pos];
	meta[pos].alpha = ptr[7*jump + pos];
	int ftype = (int)ptr[8*jump + pos];
	meta[pos].ftype = (MetaRecognition::MR_fitting_type)ftype;
	meta[pos].fitting_size = (int)ptr[9*jump + pos];
	meta[pos].translate_amount = ptr[10*jump + pos];
	meta[pos].small_score = ptr[11*jump + pos];
	meta[pos].scores_to_drop = (int)ptr[12*jump + pos];
	meta[pos].isvalid = true;
}


const char *model_to_matlab_structure(mxArray *plhs[], int num_of_feature, struct svm_model *model, int pos_plhs)
{
	int i, j, n;
	double *ptr;
	mxArray *return_model, **rhs;
	int out_id = 0;

	rhs = (mxArray **)mxMalloc(sizeof(mxArray *)*NUM_OF_RETURN_FIELD);

	// Parameters
	rhs[out_id] = mxCreateDoubleMatrix(5, 1, mxREAL);
	ptr = mxGetPr(rhs[out_id]);
	ptr[0] = model->param.svm_type;
	ptr[1] = model->param.kernel_type;
	ptr[2] = model->param.degree;
	ptr[3] = model->param.gamma;
	ptr[4] = model->param.coef0;
	out_id++;

	// nr_class
	rhs[out_id] = mxCreateDoubleMatrix(1, 1, mxREAL);
	ptr = mxGetPr(rhs[out_id]);
	ptr[0] = model->nr_class;
	out_id++;

	// total SV
	rhs[out_id] = mxCreateDoubleMatrix(1, 1, mxREAL);
	ptr = mxGetPr(rhs[out_id]);
	ptr[0] = model->l;
	out_id++;

	// openset_dim
	if (model->param.svm_type == OPENSET_OC || model->param.svm_type == ONE_WSVM || model->param.svm_type == OPENSET_BIN || 
	model->param.svm_type == OPENSET_PAIR || model->param.svm_type == ONE_VS_REST_WSVM || model->param.svm_type == PI_SVM)
	{
		rhs[out_id] = mxCreateDoubleMatrix(1, 1, mxREAL);
		ptr = mxGetPr(rhs[out_id]);
		ptr[0] = model->openset_dim;
	}
	else
	{
		rhs[out_id] = mxCreateDoubleMatrix(0, 0, mxREAL);
	}
	out_id++;	
	
	// rho
	n = model->nr_class*(model->nr_class-1)/2;
	if(model->openset_dim >0 && model->openset_dim < n ) n = model->openset_dim;
	// mexPrintf("rhosize: %d\n", n);
	rhs[out_id] = mxCreateDoubleMatrix(n, 1, mxREAL);
	ptr = mxGetPr(rhs[out_id]);
	for(i = 0; i < n; i++)
		ptr[i] = model->rho[i];
	out_id++;

	// Label
	if(model->label)
	{
		rhs[out_id] = mxCreateDoubleMatrix(model->nr_class, 1, mxREAL);
		ptr = mxGetPr(rhs[out_id]);
		for(i = 0; i < model->nr_class; i++)
			ptr[i] = model->label[i];
	}
	else
		rhs[out_id] = mxCreateDoubleMatrix(0, 0, mxREAL);
	out_id++;

	// EXTRA OPEN_SET
	if (model->param.svm_type == OPENSET_OC || model->param.svm_type == ONE_WSVM || model->param.svm_type == OPENSET_BIN || 
	model->param.svm_type == OPENSET_PAIR || model->param.svm_type == ONE_VS_REST_WSVM || model->param.svm_type == PI_SVM)
	{
		rhs[out_id] = mxCreateDoubleMatrix(1, 1, mxREAL);
		rhs[out_id+1] = mxCreateDoubleMatrix(1, 1, mxREAL);
		ptr = mxGetPr(rhs[out_id]);
		if(model->param.neg_labels)
		{
			//mexPrintf("Neg_labels 1\n");
			ptr[0] = 1;
		}
		else
		{
			ptr[0] = 0;
			ptr = mxGetPr(rhs[out_id+1]);
			ptr[0] = model->param.rejectedID;
			//mexPrintf("Neg_labels 0\n");
			//mexPrintf("Rejected_ID %d\n", model->param.rejectedID);
		}
		out_id +=2;
		
		if(model->alpha != NULL)
		{
			// mexPrintf("alpha ");
			rhs[out_id] = mxCreateDoubleMatrix(model->openset_dim, 1, mxREAL);
			ptr = mxGetPr(rhs[out_id]);
			for(int i=0; i< model->openset_dim; i++)
			{
				ptr[i] = model->alpha[i];
				// mexPrintf(" %24.20g", model->alpha[i]);
			}
			// mexPrintf("\n");
		}
		else
			rhs[out_id] = mxCreateDoubleMatrix(0, 0, mxREAL);
		out_id++;
		if(model->omega != NULL)
		{
			// mexPrintf("omega ");
			rhs[out_id] = mxCreateDoubleMatrix(model->openset_dim, 1, mxREAL);
			ptr = mxGetPr(rhs[out_id]);
			for(int i=0; i< model->openset_dim; i++)
			{
				ptr[i] = model->omega[i];
				// mexPrintf(" %24.20g", model->omega[i]);
			}
			// mexPrintf("\n");
		}
		else
			rhs[out_id] = mxCreateDoubleMatrix(0, 0, mxREAL);
		out_id++;
	}
	else
	{
		rhs[out_id] = mxCreateDoubleMatrix(0, 0, mxREAL);
		out_id++;
		rhs[out_id] = mxCreateDoubleMatrix(0, 0, mxREAL);
		out_id++;
		rhs[out_id] = mxCreateDoubleMatrix(0, 0, mxREAL);
		out_id++;
		rhs[out_id] = mxCreateDoubleMatrix(0, 0, mxREAL);
		out_id++;
	}
	
	// for ONE_VS_REST_WSVM
	if((model->param.svm_type == ONE_VS_REST_WSVM) && ((model->MRpos_one_vs_all != NULL) && (model->MRcomp_one_vs_all != NULL)))
	{
		// mexPrintf("MR_pos_one_vs_all ");
		rhs[out_id] = mxCreateDoubleMatrix(model->openset_dim, 13, mxREAL);
		ptr = mxGetPr(rhs[out_id]);
		for(int i = 0; i < model->openset_dim; i++)
			saveMR(ptr, i, model->openset_dim, model->MRpos_one_vs_all[i]);
			// mexPrintf("model->MRpos_one_vs_all[i].Save(fp)\n");
			// model->MRpos_one_vs_all[i].Save(fp);		
		out_id++;
		
		// mexPrintf("MR_comp_one_vs_all ");
		rhs[out_id] = mxCreateDoubleMatrix(model->openset_dim, 13, mxREAL);
		ptr = mxGetPr(rhs[out_id]);
		for(int i=0; i< model->openset_dim; i++)
			saveMR(ptr, i, model->openset_dim, model->MRcomp_one_vs_all[i]);
			// mexPrintf("model->MRcomp_one_vs_all[i].Save(fp)\n");
			// model->MRcomp_one_vs_all[i].Save(fp);
		out_id++;
	}
	else
	{
		rhs[out_id] = mxCreateDoubleMatrix(0, 0, mxREAL);
		out_id++;
		rhs[out_id] = mxCreateDoubleMatrix(0, 0, mxREAL);
		out_id++;
	}
	// for ONE_WSVM
	if((model->param.svm_type == ONE_WSVM) && (model->MRpos_one_class != NULL))
	{
		// mexPrintf("MR_pos_one_class ");
		rhs[out_id] = mxCreateDoubleMatrix(model->openset_dim, 13, mxREAL);
		ptr = mxGetPr(rhs[out_id]);
		for(int i=0; i< model->openset_dim; i++)
			saveMR(ptr, i, model->openset_dim, model->MRpos_one_class[i]);
			// mexPrintf("model->MRpos_one_class[i].Save(fp)\n");
			// model->MRpos_one_class[i].Save(fp);
		out_id++;
	}
	else
	{
		rhs[out_id] = mxCreateDoubleMatrix(0, 0, mxREAL);
		out_id++;
	}		
	// for PI_SVM
	if((model->param.svm_type == PI_SVM && ((model->MRpos_one_vs_all != NULL))))
	{
		// mexPrintf("MR_pos_one_vs_all ");
		rhs[10] = mxCreateDoubleMatrix(model->openset_dim, 13, mxREAL); // position 11 from 1 to 18
		ptr = mxGetPr(rhs[10]);
		for(int i=0; i< model->openset_dim; i++)
			saveMR(ptr, i, model->openset_dim, model->MRpos_one_vs_all[i]);
			// mexPrintf("model->MRpos_one_vs_all[i].Save(fp)\n");
			// model->MRpos_one_vs_all[i].Save(fp);
	}
	else
	{
		if(model->param.svm_type != ONE_VS_REST_WSVM)
			rhs[10] = mxCreateDoubleMatrix(0, 0, mxREAL);
	}
	// EXTRA OPEN_SET (END)
	
	// probA
	if(model->probA != NULL)
	{
		rhs[out_id] = mxCreateDoubleMatrix(n, 1, mxREAL);
		ptr = mxGetPr(rhs[out_id]);
		for(i = 0; i < n; i++)
			ptr[i] = model->probA[i];
	}
	else
		rhs[out_id] = mxCreateDoubleMatrix(0, 0, mxREAL);
	out_id ++;

	// probB
	if(model->probB != NULL)
	{
		rhs[out_id] = mxCreateDoubleMatrix(n, 1, mxREAL);
		ptr = mxGetPr(rhs[out_id]);
		for(i = 0; i < n; i++)
			ptr[i] = model->probB[i];
	}
	else
		rhs[out_id] = mxCreateDoubleMatrix(0, 0, mxREAL);
	out_id++;

	// nSV
	if(model->nSV)
	{
		rhs[out_id] = mxCreateDoubleMatrix(model->nr_class, 1, mxREAL);
		ptr = mxGetPr(rhs[out_id]);
		for(i = 0; i < model->nr_class; i++)
			ptr[i] = model->nSV[i];
	}
	else
		rhs[out_id] = mxCreateDoubleMatrix(0, 0, mxREAL);
	out_id++;

	// EXTRA OPEN_SET
	int limit = model->nr_class-1;
	if(model->param.svm_type == OPENSET_OC || model->param.svm_type == ONE_WSVM|| model->param.svm_type == OPENSET_BIN || model->param.svm_type == ONE_VS_REST_WSVM || model->param.svm_type == PI_SVM)
		limit = model->openset_dim;
	// EXTRA OPEN_SET (END)
	
	// sv_coef
	rhs[out_id] = mxCreateDoubleMatrix(model->l, limit, mxREAL);
	ptr = mxGetPr(rhs[out_id]);
	for(i = 0; i < limit; i++)
		for(j = 0; j < model->l; j++)
			ptr[(i*(model->l))+j] = model->sv_coef[i][j];
	out_id++;

	// SVs
	{
		int ir_index, nonzero_element;
		mwIndex *ir, *jc;
		mxArray *pprhs[1], *pplhs[1];	

		if(model->param.kernel_type == PRECOMPUTED)
		{
			nonzero_element = model->l;
			num_of_feature = 1;
		}
		else
		{
			nonzero_element = 0;
			for(i = 0; i < model->l; i++) {
				j = 0;
				while(model->SV[i][j].index != -1) 
				{
					nonzero_element++;
					j++;
				}
			}
		}

		// SV in column, easier accessing
		rhs[out_id] = mxCreateSparse(num_of_feature, model->l, nonzero_element, mxREAL);
		ir = mxGetIr(rhs[out_id]);
		jc = mxGetJc(rhs[out_id]);
		ptr = mxGetPr(rhs[out_id]);
		jc[0] = ir_index = 0;		
		for(i = 0;i < model->l; i++)
		{
			if(model->param.kernel_type == PRECOMPUTED)
			{
				// make a (1 x model->l) matrix
				ir[ir_index] = 0; 
				ptr[ir_index] = model->SV[i][0].value;
				ir_index++;
				jc[i+1] = jc[i] + 1;
			}
			else
			{
				int x_index = 0;
				while (model->SV[i][x_index].index != -1)
				{
					ir[ir_index] = model->SV[i][x_index].index - 1; 
					ptr[ir_index] = model->SV[i][x_index].value;
					ir_index++, x_index++;
				}
				jc[i+1] = jc[i] + x_index;
			}
		}
		// transpose back to SV in row
		pprhs[0] = rhs[out_id];
		if(mexCallMATLAB(1, pplhs, 1, pprhs, "transpose"))
			return "cannot transpose SV matrix";
		rhs[out_id] = pplhs[0];
		out_id++;
	}

	/* Create a struct matrix contains NUM_OF_RETURN_FIELD fields */
	return_model = mxCreateStructMatrix(1, 1, NUM_OF_RETURN_FIELD, field_names);

	/* Fill struct matrix with input arguments */
	for(i = 0; i < NUM_OF_RETURN_FIELD; i++)
		mxSetField(return_model,0,field_names[i],mxDuplicateArray(rhs[i]));
	/* return */
	plhs[pos_plhs] = return_model;
	mxFree(rhs);

	return NULL;
}

struct svm_model *matlab_matrix_to_model(const mxArray *matlab_struct, const char **msg)
{
	int i, j, n, num_of_fields;
	double *ptr;
	int id = 0;
	struct svm_node *x_space;
	struct svm_model *model;
	mxArray **rhs;

	num_of_fields = mxGetNumberOfFields(matlab_struct);
	if(num_of_fields != NUM_OF_RETURN_FIELD) 
	{
		*msg = "number of return field is not correct";
		return NULL;
	}
	rhs = (mxArray **) mxMalloc(sizeof(mxArray *)*num_of_fields);

	for(i=0;i<num_of_fields;i++)
		rhs[i] = mxGetFieldByNumber(matlab_struct, 0, i);

	model = Malloc(struct svm_model, 1);
	model->rho = NULL;
	model->probA = NULL;
	model->probB = NULL;
	model->label = NULL;
	model->nSV = NULL;
	model->free_sv = 1; // XXX
	
	// EXTRA OPEN_SET
	model->alpha = NULL;
    model->omega = NULL;
	model->MRpos_one_vs_all = NULL;
	model->MRcomp_one_vs_all = NULL;  
	model->MRpos_one_class = NULL;

    model->openset_dim = 0;
    model->param.rejectedID = -9999; /* id for rejected classes (-99999 is the default) */
    model->param.neg_labels = false;
	// EXTRA OPEN_SET (END)
	
	ptr = mxGetPr(rhs[id]);
	model->param.svm_type = (int)ptr[0];
	model->param.kernel_type  = (int)ptr[1];
	model->param.degree	  = (int)ptr[2];
	model->param.gamma	  = ptr[3];
	model->param.coef0	  = ptr[4];
	id++;

	ptr = mxGetPr(rhs[id]);
	model->nr_class = (int)ptr[0];
	id++;

	ptr = mxGetPr(rhs[id]);
	model->l = (int)ptr[0];
	id++;

	// EXTRA OPEN_SET
	if(mxIsEmpty(rhs[id]) == 0)
	{
		ptr = mxGetPr(rhs[id]);
		model->openset_dim = (int)ptr[0];
		// mexPrintf("model->openset_dim: %d\n", model->openset_dim);
	}
	id++;
	// EXTRA OPEN_SET (END)
	
	// rho
	n = model->nr_class * (model->nr_class-1)/2;
	// EXTRA OPEN_SET
	if(model->openset_dim > 0 && model->openset_dim < n )
		n = model->openset_dim;
	// EXTRA OPEN_SET (END)
	model->rho = (double*) malloc(n*sizeof(double));
	ptr = mxGetPr(rhs[id]);
	for(i = 0; i < n; i++)
		model->rho[i] = ptr[i];
	id++;

	// label
	if(mxIsEmpty(rhs[id]) == 0)
	{
		model->label = (int*) malloc(model->nr_class*sizeof(int));
		ptr = mxGetPr(rhs[id]);
		for(i=0;i<model->nr_class;i++)
			model->label[i] = (int)ptr[i];
	}
	id++;

	// EXTRA OPEN_SET
	// -> Neg_labels
	model->param.neg_labels = false;
	if(mxIsEmpty(rhs[id]) == 0)
	{
		ptr = mxGetPr(rhs[id]);
		int junk = (int)ptr[0];
		if(junk > 0)
			model->param.neg_labels = true;
		// mexPrintf("neg_labels: %d\n", model->param.neg_labels);
	}
	id++;
	
	// -> Rejected_ID
	if(mxIsEmpty(rhs[id]) == 0)
	{
		ptr = mxGetPr(rhs[id]);
		model->param.rejectedID = (int)ptr[0];
		// mexPrintf("rejectedID: %d\n", model->param.rejectedID);
	}
	id++;
	
	// -> alpha
	if(mxIsEmpty(rhs[id]) == 0)
	{
		ptr = mxGetPr(rhs[id]);
		int n = model->openset_dim;
		if(n > 0)
		{
			model->alpha = (double*) malloc(n*sizeof(double));
			for(int i = 0; i < n; i++) 
				model->alpha[i] = (double)ptr[i];
		}
		else
		{
			mexPrintf("Openset alpha found for non openset class, ignoring it.\n");
			model->alpha = (double*) malloc(1*sizeof(double));
			model->alpha[0] = 0;
		}
	}
	id++;
	
	// -> omega
	if(mxIsEmpty(rhs[id]) == 0)
	{
		ptr = mxGetPr(rhs[id]);
		int n = model->openset_dim;
		if(n > 0)
		{
			model->omega = (double*) malloc(n*sizeof(double));
			for(int i = 0; i < n; i++) 
				model->omega[i] = (double)ptr[i];
		}
		else
		{
			mexPrintf("Openset omega found for non openset class, ignoring it.\n");
			model->omega = (double*) malloc(1*sizeof(double));
			model->omega[0] = 0;
		}		
	}
	id++;
	
	// -> for ONE_VS_REST_WSVM POSITIVE
	if(mxIsEmpty(rhs[id]) == 0)
	{
		int n = model->openset_dim;
		if(n > 0 && (model->param.svm_type == ONE_VS_REST_WSVM) || (model->param.svm_type == PI_SVM))
		{
			model->MRpos_one_vs_all = new MetaRecognition[model->openset_dim];
			ptr = mxGetPr(rhs[id]);
			for(int i = 0; i < n; i++)
			{
				loadMR(ptr, i, n, model->MRpos_one_vs_all);
				//model->MRpos_one_vs_all[i].Load(fp);
			}
		}
		else
		{
			mexPrintf("Openset MR_pos_one_vs_all found for non ONE_VS_REST WSVM class, ignoring it.\n");
			model->MRpos_one_vs_all = (MetaRecognition*) malloc(1*sizeof(MetaRecognition));
			// model->MRpos_one_vs_all[0] = 0;
		}
	}
	id++;
	
	// -> for ONE_VS_REST_WSVM COMPLIMENT
	if(mxIsEmpty(rhs[id]) == 0)
	{
		int n = model->openset_dim;
		if(n > 0 && (model->param.svm_type == ONE_VS_REST_WSVM))
		{
			model->MRcomp_one_vs_all = new MetaRecognition[model->openset_dim];
			ptr = mxGetPr(rhs[id]);
			for(int i = 0; i < n; i++)
			{
				loadMR(ptr, i, n, model->MRcomp_one_vs_all);
				// model->MRcomp_one_vs_all[i].Load(fp);
			}
		}
		else
		{
			mexPrintf("Openset MR_comp_one_vs_all found for non ONE_VS_REST WSVM class, ignoring it.\n");
			model->MRcomp_one_vs_all = (MetaRecognition*) malloc(1*sizeof(MetaRecognition));
			model->MRcomp_one_vs_all[0] = 0;
		}
	}
	id++;
	
	// -> for ONE_WSVM POSITIVE
	if(mxIsEmpty(rhs[id]) == 0)
	{
		int n = model->openset_dim;
		if(n > 0 && (model->param.svm_type == ONE_WSVM))
		{
			model->MRpos_one_class = new MetaRecognition[model->openset_dim];
			ptr = mxGetPr(rhs[id]);
			for(int i=0; i < n; i++)
			{
				loadMR(ptr, i, n, model->MRpos_one_class);
				// model->MRpos_one_class[i].Load(fp);
			}
		}
		else
		{
			mexPrintf("Openset MRpos_one_class found for non ONE_WSVM class, ignoring it.\n");
			model->MRpos_one_class = (MetaRecognition*) malloc(1*sizeof(MetaRecognition));
			model->MRpos_one_class[0] = 0;
		}
	}
	id++;
	// EXTRA OPEN_SET (END)
	
	// probA
	if(mxIsEmpty(rhs[id]) == 0)
	{
		model->probA = (double*) malloc(n*sizeof(double));
		ptr = mxGetPr(rhs[id]);
		for(i=0;i<n;i++)
			model->probA[i] = ptr[i];
	}
	id++;

	// probB
	if(mxIsEmpty(rhs[id]) == 0)
	{
		model->probB = (double*) malloc(n*sizeof(double));
		ptr = mxGetPr(rhs[id]);
		for(i=0;i<n;i++)
			model->probB[i] = ptr[i];
	}
	id++;

	// nSV
	if(mxIsEmpty(rhs[id]) == 0)
	{
		model->nSV = (int*) malloc(model->nr_class*sizeof(int));
		ptr = mxGetPr(rhs[id]);
		for(i=0;i<model->nr_class;i++)
			model->nSV[i] = (int)ptr[i];
	}
	id++;

	// sv_coef
	// EXTRA OPEN_SET
	int m = model->nr_class - 1;
	if(model->param.svm_type == OPENSET_OC || model->param.svm_type == ONE_WSVM || model->param.svm_type == OPENSET_BIN || model->param.svm_type == ONE_VS_REST_WSVM || model->param.svm_type == PI_SVM)
		m++;
	ptr = mxGetPr(rhs[id]);
	model->sv_coef = (double**) malloc(m*sizeof(double));
	for(i = 0 ; i < m; i++)
		model->sv_coef[i] = (double*) malloc((model->l)*sizeof(double));
	for(i = 0; i < m; i++)
		for(j = 0; j < model->l; j++)
			model->sv_coef[i][j] = ptr[i*(model->l)+j];
	id++;
	// EXTRA OPEN_SET (END)

	// SV
	{
		int sr, sc, elements;
		int num_samples;
		mwIndex *ir, *jc;
		mxArray *pprhs[1], *pplhs[1];

		// transpose SV
		pprhs[0] = rhs[id];
		if(mexCallMATLAB(1, pplhs, 1, pprhs, "transpose")) 
		{
			svm_free_and_destroy_model(&model);
			*msg = "cannot transpose SV matrix";
			return NULL;
		}
		rhs[id] = pplhs[0];

		sr = (int)mxGetN(rhs[id]);
		sc = (int)mxGetM(rhs[id]);

		ptr = mxGetPr(rhs[id]);
		ir = mxGetIr(rhs[id]);
		jc = mxGetJc(rhs[id]);

		num_samples = (int)mxGetNzmax(rhs[id]);

		elements = num_samples + sr;

		model->SV = (struct svm_node **) malloc(sr * sizeof(struct svm_node *));
		x_space = (struct svm_node *)malloc(elements * sizeof(struct svm_node));

		// SV is in column
		for(i=0;i<sr;i++)
		{
			int low = (int)jc[i], high = (int)jc[i+1];
			int x_index = 0;
			model->SV[i] = &x_space[low+i];
			for(j=low;j<high;j++)
			{
				model->SV[i][x_index].index = (int)ir[j] + 1; 
				model->SV[i][x_index].value = ptr[j];
				x_index++;
			}
			model->SV[i][x_index].index = -1;
		}

		id++;
	}
	mxFree(rhs);

	return model;
}

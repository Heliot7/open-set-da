#ifndef _LIBSVM_H
#define _LIBSVM_H

#define LIBSVM_VERSION 310

#define USEWSVM

#ifdef USEWSVM
#include "libMR/MetaRecognition.h"
#endif


#ifdef __cplusplus
extern "C" {
#endif

extern int libsvm_version;

struct svm_node{
	int index;
	double value;
};

struct svm_problem{
  int l;
  double *y;
  struct svm_node **x;
  int nr_classes;
  int *labels;
};


  typedef enum { C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR, OPENSET_OC, OPENSET_PAIR, OPENSET_BIN, ONE_VS_REST_WSVM, ONE_WSVM, PI_SVM} svm_type_t;	/* svm_type */
  typedef enum { LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED } kernel_t; /* kernel_type */

  typedef enum {OPT_PRECISION, OPT_RECALL,  OPT_FMEASURE,  OPT_HINGE, OPT_BALANCEDRISK}  openset_optimization_t;



struct svm_parameter{
	int svm_type;
	int kernel_type;
	int do_open;	/* do we want to do open-set expansion of base kernel */
	int degree;	/* for poly */
	double gamma;	/* for poly/rbf/sigmoid */
	double coef0;	/* for poly/sigmoid */
	/* these are for training only */
	double cache_size; /* in MB */
	double eps;	/* stopping criteria */
	double C;	/* for C_SVC, EPSILON_SVR and NU_SVR */
	int nr_weight;		/* for C_SVC */
	int nr_fold;		/* for cross-validation in training */
	int cross_validation;		/* for cross-validation */
	int *weight_label;	/* for C_SVC */
	double* weight;		/* for C_SVC */
	double nu;	/* for NU_SVC, ONE_CLASS, and NU_SVR */
	double p;	/* for EPSILON_SVR */
	int shrinking;	/* use the shrinking heuristics */
	int probability; /* do probability estimates */
	bool neg_labels; /* do we consider negative class labels (like -1) as a label for openset, or just negative.. default is false */
	bool exaustive_open; /* do we do exaustive optimization for openset.. default is false */
    openset_optimization_t optimize; /* choice of what to optimize */
    double beta; /* for use in f-measure optimization */
    double near_preasure, far_preasure; /* for openset risk preasures */
    double openset_min_probability; /* for WSVM openset, what is minimum probability to consider positive */
    double openset_min_probability_one_wsvm; /* for WSVM openset, what is minimum probability to consider positive for one class wsvm */
    FILE* vfile; /* for logging verbose stuff during debugging */
    int  rejectedID; /* id for rejected classes (-99999 is the default) */
    double cap_cost;	/* for C_SVC, EPSILON_SVR and NU_SVR */
    double cap_gamma;	/* for poly/rbf/sigmoid */
};

//
// svm_model
// 
struct svm_model{
	struct svm_parameter param;	/* parameter */
	int nr_class;		/* number of classes, = 2 in regression/one class svm */
	int l;			/* total #SV */
	struct svm_node **SV;		/* SVs (SV[l]) */
	double **sv_coef;	/* coefficients for SVs in decision functions (sv_coef[k-1][l]) */
	double *rho;		/* constants in decision functions (rho[k*(k-1)/2]) */
	double *probA;		/* pariwise probability information */
	double *probB;
    int openset_dim;        /* dimension of data for 1-vs-set models,  if open_set, wsvm or open_bset (5,6,7) then openset_dim=k, if open_pair then its k*(k-1)/2*/
    double *alpha, *omega;  /* planes offsets for 1-vs-set   alpha[openset_dim], omega[openset_dim] */
    //double *wbltrans,*wblshape,*wblscale;        /* weibul parms for wsvm   all of dimension [openset_dim] */
	#ifdef USEWSVM
		MetaRecognition *MRpos_one_vs_all, *MRcomp_one_vs_all;   //MetaRecognition Objects for positive (inclass) and complement classifiers for 1-vs-all
        MetaRecognition *MRpos_one_class;	//MetaRecognition Objects for positive (inclass) one-class
	#endif
	/* for classification only */

	int *label;		/* label of each class (label[k]) */
	int *nSV;		/* number of SVs for each class (nSV[k]) */
				/* nSV[0] + nSV[1] + ... + nSV[k-1] = l */
	/* XXX */
	int free_sv;		/* 1 if svm_model is created by svm_load_model and needs to free its memory*/
				/* 0 if svm_model is created by svm_train */
};

struct svm_model *svm_train(const struct svm_problem *prob, const struct svm_parameter *param);
void svm_cross_validation(const struct svm_problem *prob, const struct svm_parameter *param, int nr_fold, double *target);
void svm_cross_validation_wsvm(const struct svm_problem *prob, const struct svm_parameter *param,const struct svm_problem *prob_one_wsvm, const struct svm_parameter *param_one_wsvm, int nr_fold, double *target);

int svm_save_model(const char *model_file_name, const struct svm_model *model);
struct svm_model *svm_load_model(const char *model_file_name);

int svm_get_svm_type(const struct svm_model *model);
int svm_get_nr_class(const struct svm_model *model);
void svm_get_labels(const struct svm_model *model, int *label);
double svm_get_svr_probability(const struct svm_model *model);
double svm_predict_values(const struct svm_model *model, const struct svm_node *x, double* dec_values);
double svm_predict_values_extended(const struct svm_model *model, const struct svm_node *x, 
								   double*& dec_values, double **&scores, int*& vote);
double svm_predict_values_extended_plus_one_wsvm(const struct svm_model *model,const struct svm_model *model_one_wsvm, const struct svm_node *x, 
								   double*& dec_values_wsvm,double*& dec_values_one_wsvm, double **&scores, int*& vote);

double svm_predict(const struct svm_model *model, const struct svm_node *x);
double svm_predict_extended(const struct svm_model *model, const struct svm_node *x,
							double **&scores, int *&vote);
double svm_predict_extended_plus_one_wsvm(const struct svm_model *model,const struct svm_model *model_one_wsvm, const struct svm_node *x,
						    double **&scores, int *&vote);
double svm_predict_probability(const struct svm_model *model, const struct svm_node *x, double* prob_estimates);

void svm_free_model_content(struct svm_model *model_ptr);
void svm_free_and_destroy_model(struct svm_model **model_ptr_ptr);
void svm_destroy_param(struct svm_parameter *param);

const char *svm_check_parameter(const struct svm_problem *prob, const struct svm_parameter *param);
int svm_check_probability_model(const struct svm_model *model);

void svm_set_print_string_function(void (*print_func)(const char *));

  typedef  unsigned long ulong;
#ifdef __cplusplus
}
#endif

#endif /* _LIBSVM_H */

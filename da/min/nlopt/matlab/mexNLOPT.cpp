#include <ctime>
#include <vector>
#include <math.h>

#include <Eigen/Dense>
#include <Eigen/Core>

#include "mex.h"
#include "nlopt.hpp"

int counter = 0;

using namespace std;
using namespace Eigen;
using Eigen::MatrixXd;

struct mFuncData
{
	int numDims, numSamples;
	MatrixXd X, Y;
	MatrixXd XX, YX;
};

double myfunc(unsigned n, const double *inW, double *grad, void *inData)
{
	++counter;

	mFuncData* data = (mFuncData*)inData;
	MatrixXd W = Map<MatrixXd>((double*)inW, data->numDims, data->numDims);

	// 2 * W*(X*X') - Y*X' - Y*X';
	if (grad)
	{
		clock_t beginM = clock();

		mwSize size[2]; size[0] = data->numDims;  size[1] = data->numDims;
		mxArray* prhs[2];
		prhs[0] = mxCreateNumericArray(2, size, mxDOUBLE_CLASS, mxREAL);
		double* WMatlab = (double*)mxGetData(prhs[0]);
		Map<MatrixXd>(WMatlab, data->numDims, data->numDims) = W;
		prhs[1] = mxCreateNumericArray(2, size, mxDOUBLE_CLASS, mxREAL);
		double* XMatlab = (double*)mxGetData(prhs[1]);
		Map<MatrixXd>(XMatlab, data->numDims, data->numDims) = data->XX;

		mxArray* plhs[1];
		int output = mexCallMATLAB(1, plhs, 2, prhs, "mtimes");
		double* res = (double*)mxGetData(plhs[0]);
		MatrixXd newW = Map<MatrixXd>(res, data->numDims, data->numDims);
		MatrixXd gradient = 2.0*newW - data->YX - data->YX;

		Map<MatrixXd>(grad, data->numDims, data->numDims) = gradient;

		clock_t endM = clock();
		// mexPrintf("time step newgrad: %f\n", difftime(endM, beginM));
	}
	

	
	// val = norm(W*X - Y);
	clock_t begin1 = clock();
	MatrixXd res = W*data->X - data->Y;
	clock_t end1 = clock();
	//mexPrintf("time step main: %f\n", difftime(end1, begin1));

	// Matlab call
	clock_t beginM = clock();
	mxArray* prhs[1];
	mwSize size[2]; size[0] = data->numDims;  size[1] = data->numSamples;
	prhs[0] = mxCreateNumericArray(2, size, mxDOUBLE_CLASS, mxREAL);
	double* resMatlab = (double*)mxGetData(prhs[0]);
	Map<MatrixXd>(resMatlab, data->numDims, data->numSamples) = res;
	mxArray* plhs[1];
	mexCallMATLAB(1, plhs, 1, prhs, "norm");
	double norm = (double)mxGetScalar(plhs[0]);
	clock_t endM = clock();
	// mexPrintf("time step norm: %f\n", difftime(endM, beginM));
	mexPrintf("[iter %d] energy: %f\n", counter, norm);

	/*
	clock_t begin2 = clock();
	double energy = res.operatorNorm();
	clock_t end2 = clock();
	mexPrintf("time step norm: %f\n", difftime(end2, begin2));
	mexPrintf("[iter %d] energy: %f\n", counter, energy);
	return energy;
	*/

	return norm;
}

typedef struct
{
	double a, b;
} mConstraintData;

double myconstraint(unsigned n, const double *x, double *grad, void *data)
{
	mConstraintData *d = (mConstraintData *)data;
	double a = d->a, b = d->b;
	if (grad) {
		grad[0] = 3 * a * (a*x[0] + b) * (a*x[0] + b);
		grad[1] = -1.0;
	}
	return ((a*x[0] + b) * (a*x[0] + b) * (a*x[0] + b) - x[1]);
}

void run_optimiser(double* x, double* residual, int numDims, int numSamples, double* srcCentres, double* tgtCentres)
{
	nlopt_opt opt;
	
	opt = nlopt_create(NLOPT_LD_MMA, numDims*numDims); /* algorithm and dimensionality */
	// opt = nlopt_create(NLOPT_LN_COBYLA, numDims*numDims); /* algorithm and dimensionality */

	mFuncData funcData;
	funcData.numDims = numDims;
	funcData.numSamples = numSamples;
	funcData.X = Map<MatrixXd>(srcCentres, numDims, numSamples);
	funcData.Y = Map<MatrixXd>(tgtCentres, numDims, numSamples);

	// Precomputation
	funcData.XX = funcData.X * funcData.X.transpose();
	funcData.YX = funcData.Y * funcData.X.transpose();
	
	nlopt_set_min_objective(opt, myfunc, &funcData);
	nlopt_set_xtol_rel(opt, 1e-4);
	nlopt_set_maxeval(opt, 25);

	// Constraints:
	//mConstraintData data[2] = { { 2, 0 }, { -1, 1 } };
	//nlopt_add_inequality_constraint(opt, myconstraint, &data[0], 1e-4);
	//nlopt_add_inequality_constraint(opt, myconstraint, &data[1], 1e-4);

	if (nlopt_optimize(opt, x, residual) < 0)
		printf("nlopt failed!\n");
	else
		printf("found minimum after %d evaluations with residual %f\n", counter, *residual);

	nlopt_destroy(opt);
}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	/* Macros for the input arguments */
	#define srcCentres_IN prhs[0]
	#define tgtCentres_IN prhs[1]

	/* Macros for the output arguments */
	#define x_OUT plhs[0]
	#define residual_OUT plhs[1]

	/* Check correctness of input/output arguments */
	if (nrhs < 2 || nrhs > 2)
		mexErrMsgTxt("Wrong number of input arguments.");
	else if (nlhs > 2)
		mexErrMsgTxt("Too many output arguments.");

	/* Get input data */
	double* srcCentres = (double*)mxGetData(srcCentres_IN);
	double* tgtCentres = (double*)mxGetData(tgtCentres_IN);
	int numDims = (int)mxGetM(srcCentres_IN);
	int numSamples = (int)mxGetN(srcCentres_IN);

	/* Create output data */
	x_OUT = mxCreateNumericMatrix(numDims, numDims, mxDOUBLE_CLASS, mxREAL);
	double* x = (double*)mxGetData(x_OUT);
	residual_OUT = mxCreateNumericMatrix(1, 1, mxDOUBLE_CLASS, mxREAL);
	double* residual = (double*)mxGetData(residual_OUT);

	// Initial guess (Id matrix)
	for (int idx = 0; idx < numDims*numDims; ++idx)
	{
		if (idx % (numDims + 1) == 0)
			x[idx] = 1.0;
		else
			x[idx] = 0.0;
	}
	
	// Reset counter
	counter = 0;

	/* Call method */
	run_optimiser(x, residual, numDims, numSamples, srcCentres, tgtCentres);

	return;
}

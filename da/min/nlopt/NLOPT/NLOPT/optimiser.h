#include <iostream>
#include <vector>
#include <math.h>

#include <Eigen/Dense>
#include <Eigen/Core>

#include "nlopt.hpp"

int counter = 0;

using namespace std;
using namespace Eigen;
using Eigen::MatrixXd;
using Eigen::VectorXd;

typedef struct
{
	int numDims, numSamples;
	double *srcCentres, *srcAssign;
	double *tgtCentres, *tgtAssign;
	//MatrixXd srcCentres, tgtCentres;
	//VectorXd srcAssign, tgtAssign;
} mFuncData;

double myfunc(unsigned n, const double *dataW, double *grad, void *data)
{
	++counter;

	mFuncData* funcData = (mFuncData*)data;
	int numDims = funcData->numDims;
	int numSamples = funcData->numSamples;

	double* X = funcData->srcCentres;
	double* srcAssign = funcData->srcAssign;
	double* Y = funcData->tgtCentres;
	double* tgtAssign = funcData->tgtAssign;

	//MatrixXd eigenX = funcData->srcCentres;
	//VectorXd eigenSrcAssign = funcData->srcAssign;
	//MatrixXd eigenY = funcData->tgtCentres;
	//VectorXd eigenTgtAssign = funcData->tgtAssign;
	
	const double* W = dataW;

	MatrixXd eigenX = Map<MatrixXd>(X, numSamples, numDims);
	MatrixXd eigenY = Map<MatrixXd>(Y, numSamples, numDims);
	MatrixXd eigenW = Map<MatrixXd>((double*)W, numDims, numDims);

	if (grad)
	{
		/*
		double* auxXX = (double*)malloc(numDims*numDims*sizeof(*auxXX));
		double* auxYX = (double*)malloc(numDims*numDims*sizeof(*auxYX));
		double sum = 0.0;

		// X'*X
		for (int i = 0; i < numDims; ++i)
			for (int j = 0; j < numDims; ++j)
			{
				sum = 0.0;
				for (int s = 0; s < numSamples; ++s)
					sum += X[j + numDims*(int)srcAssign[s]] * X[(int)srcAssign[s] + numSamples*i];
				auxXX[j + numDims*i] = sum;
			}
		//cout << "X'*X:" << endl;
		//for (int i = 0; i < numDims*numDims; ++i)
			//cout << auxXX[i] << " ";
		//cout << endl;

		// W*(X'*X)
		//cout << "W*(X'*X):" << endl;
		for (int i = 0; i < numDims; ++i)
			for (int j = 0; j < numDims; ++j)
			{
				grad[j + numDims*i] = 0.0;
				for (int dim = 0; dim < numDims; ++dim)
					grad[j + numDims*i] += W[j + numDims*dim] * auxXX[dim + numDims*i];
				//cout << grad[j + numDims*i] << " ";
			}
		//cout << endl;

		// Y'*X
		//cout << "Y'*X:" << endl;
		for (int i = 0; i < numDims; ++i)
			for (int j = 0; j < numDims; ++j)
			{
				sum = 0.0;
				for (int s = 0; s < numSamples; ++s)
					sum += Y[j + numDims*(int)tgtAssign[s]] * X[(int)srcAssign[s] + numSamples*i];
				auxYX[j + numDims*i] = sum;
				//cout << auxYX[j + numDims*i] << " ";
			}
		//cout << endl;

		// gradient = 2*W*(X'*X) - Y'*X - Y'*X;
		//cout << "gradient:" << endl;
		for (int i = 0; i < numDims; ++i)
			for (int j = 0; j < numDims; ++j)
			{
				grad[j + numDims*i] = 2.0 * grad[j + numDims*i] - auxYX[j + numDims*i] - auxYX[j + numDims*i];
				//cout << grad[j + numDims*i] << " ";
			}
		//cout << endl;

		free(auxYX);
		free(auxXX);
		*/

		MatrixXd gradient = 2 * eigenW * (eigenX.transpose() * eigenX) - (eigenY.transpose() * eigenX) - (eigenY.transpose() * eigenX);
		//cout << "X: " << endl << eigenX << endl;
		//cout << "Y: " << endl << eigenY << endl;
		//cout << "W: " << endl << eigenW << endl;
		//cout << "X'*X: " << endl << (eigenX.transpose() * eigenX) << endl;
		//cout << "Y'*X: " << endl << (eigenY.transpose() * eigenX) << endl;
		//cout << "W*(X'*X): " << endl << eigenW * (eigenX.transpose() * eigenX) << endl;
		// cout << gradient << endl;
		grad = gradient.data();
		//cout << endl;
		//for (int i = 0; i < numDims*numDims; ++i)
		//	cout << grad[i] << " ";
		//cout << endl;
	}

	// val = norm(W*X - Y);
	// 1 -> numSamples
	double val = 0.0;
	double sum = 0.0;
	/*
	cout << endl << "[it " << counter << "] Val: " << endl;
	for (int idxX = 0; idxX < numSamples; ++idxX)
	{
		val = 0.0;
		for (int i = 0; i < numDims; ++i)
		{
			double mul = 0.0;
			int idxSample = (int)srcAssign[idxX];
			for (int j = 0; j < numDims; ++j)
			{
				mul += X[idxSample + numSamples*j] * W[j + numDims*i];
			}
			idxSample = (int)tgtAssign[idxX];
			val += (mul - Y[idxSample + numSamples*i])*(mul - Y[idxSample + numSamples*i]);
			cout << mul - Y[idxSample + numSamples*i] << " ";
		}
		sum += sqrt(val);
		cout << endl;
	}
	cout << sum << endl;
	*/

	MatrixXd res = eigenX*eigenW - eigenY;
	//cout << endl << res << endl;
	//cout << res.operatorNorm() << endl;
	
	sum = res.operatorNorm();
	return sum;
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

void run_optimiser(double* x, double* residual, int numDims, int numSamples, double* srcCentres, double* srcAssign, double* tgtCentres, double* tgtAssign)
{
	nlopt_opt opt;
	opt = nlopt_create(NLOPT_LD_MMA, numDims*numDims); /* algorithm and dimensionality */
	// opt = nlopt_create(NLOPT_LN_COBYLA, numDims*numDims); /* algorithm and dimensionality */

	mFuncData funcData;
	funcData.numDims = numDims; funcData.numSamples = numSamples;

	funcData.srcCentres = srcCentres; funcData.srcAssign = srcAssign;
	funcData.tgtCentres = tgtCentres; funcData.tgtAssign = tgtAssign;

	
	//funcData.srcCentres = Map<MatrixXd>(srcCentres, numSamples, numDims);
	//funcData.srcAssign = Map<VectorXd>(srcAssign, numSamples);
	//funcData.tgtCentres = Map<MatrixXd>(tgtCentres, numSamples, numDims);
	//funcData.tgtAssign = Map<VectorXd>(tgtAssign, numSamples);

	nlopt_set_min_objective(opt, myfunc, &funcData);
	nlopt_set_xtol_rel(opt, 1e-4);
	nlopt_set_maxeval(opt, 1000);

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
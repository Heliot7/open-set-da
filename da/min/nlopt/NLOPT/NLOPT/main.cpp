#include <iostream>
#include "optimiser.h"

int main(int argc, char* argv[])
{
	// Initialisation
	int numDims = 3;
	int numSamples = 2;
	int numClusters = 2;
	double* srcCentres = new double[numClusters*numDims];
	for (int idx = 0; idx < numClusters*numDims; ++idx)
		srcCentres[idx] = 1.0;
	double* srcAssign = new double[numSamples];
	srcAssign[0] = 0; srcAssign[1] = 1;
	double* tgtCentres = new double[numClusters*numDims];
	for (int idx = 0; idx < numClusters*numDims; ++idx)
		tgtCentres[idx] = 2.0;
	double* tgtAssign = new double[numSamples];
	tgtAssign[0] = 0; tgtAssign[1] = 1;
	double* x = new double[numDims*numDims];
	for (int idx = 0; idx < numDims*numDims; ++idx)
	{
		if (idx % (numDims + 1) == 0)
			x[idx] = 1.0;
		else
			x[idx] = 0.0;
	}
	double residual;

	// Run optimisation process
	run_optimiser(x, &residual, numDims, numSamples, srcCentres, srcAssign, tgtCentres, tgtAssign);

	cout << endl;
	for (int idx = 0; idx < numDims*numDims; ++idx)
		cout << x[idx] << " ";
	cout << endl;

	delete x;
	delete srcCentres;
	delete srcAssign;
	delete tgtCentres;
	delete tgtAssign;

	return 0;
}

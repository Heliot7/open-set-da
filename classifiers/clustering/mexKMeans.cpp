#include <iostream>
#include <stdio.h>
#include <algorithm>
#include <math.h>
#include <cstring>
#include <limits>
#include <vector>

#include "mex.h"

using namespace std;

void kMeans(double* centroids, double* labels, int numSamples, int numDims, double* features, double* ar, int K, int numIter)
{
    // Several runs and stick with the values of the iteration with the lowest energy score
    vector<double> bestCentroids; bestCentroids.resize(K*numDims);
    vector<double> bestLabels; bestLabels.resize(numSamples);
    double bestEnergy = numeric_limits<double>::max();
    mexPrintf("Run K-Means\n");
    for(unsigned globalIt = 0; globalIt < numIter; ++globalIt)
    {
        // mexPrintf(" %d", globalIt);

        // Initialise centroids with samples that are equitatively separated in the features array
        // int step = (int)(numSamples / K);
        for(int c = 0; c < K; ++c)
        {
            int randFeature = rand() % numSamples;
            for(int dim = 0; dim < numDims; ++dim)
            {
                // double value = rand()/RAND_MAX*2.0 - 1.0;
                // mexPrintf("%f\n",value);
                // centroids[dim*K + c] = value;
                // centroids[dim*K + c] = features[dim*numSamples + step*c];
                centroids[dim*K + c] = features[dim*numSamples + randFeature];
            }
        }

        for(int sample = 0; sample < numSamples; ++sample)
            labels[sample] = 0.0;

        // Run K-means
        vector<double> sumCentroid; sumCentroid.resize(K*numDims);
        vector<double> arCentroid; arCentroid.resize(K);
        vector<int> counter; counter.resize(K);
        int innerIt = 1000;
        double energy = 0.0;
        for(int it = 0; it < innerIt; ++it)
        {
            // mexPrintf("Iteration %d", it+1);

            // Reset sum and numbers of centroids per sample
            for(int c = 0; c < K; ++c)
            {
                for(int dim = 0; dim < numDims; ++dim)
                    sumCentroid[dim*K + c] = 0.0;
                arCentroid[c] = 0.0;
                counter[c] = 0;
            }
            int updates = 0;
            energy = 0.0;

            // Assign a centroid per feature sample.
            // -> For all samples
            for(int sample = 0; sample < numSamples; ++sample)
            {
                double minDist = numeric_limits<double>::infinity();
                int mCentroid = 0;

                // -> Check all centroids
                for(int c = 0; c < K; ++c)
                {
                    double dist = 0.0;
                    // Measure eucladian distance + weighted AR diff
                    for(int dim = 0; dim < numDims; ++dim)
                        dist += (centroids[dim*K + c] - features[dim*numSamples + sample]) *
                                (centroids[dim*K + c] - features[dim*numSamples + sample]);
                    // dist = 0.9*dist + 0.1*abs(arCentroid[c] - ar[sample]);
                    // dist = sqrt(dist);

                    if(dist < minDist)
                    {
                        mCentroid = c;
                        minDist = dist;
                    }
                }
                energy += minDist;

                if(labels[sample] != mCentroid + 1)
                        updates++;

                labels[sample] = mCentroid + 1;

                // Recalculate centroids
                counter[mCentroid]++;
                for(int dim = 0; dim < numDims; ++dim)
                    sumCentroid[dim*K + mCentroid] += features[dim*numSamples + sample];
                // arCentroid[mCentroid] += ar[sample];
            }

            // Update centroids
            for(int c = 0; c < K; ++c)
                if(counter[c] != 0)
                {
                    for(int dim = 0; dim < numDims; ++dim)
                        centroids[dim*K + c] = sumCentroid[dim*K + c] / counter[c];
                    arCentroid[c] /= counter[c];
                }

            // Print energy
            // mexPrintf(" - energy: %f ", energy);

            // Check if we can leave it like this...
            // mexPrintf(" - %d udpates\n", updates);
            if(updates < 0.05*numSamples)
            {
                // mexPrintf("Finished with %d iterations... Th below 0.05*numSamples\n", it);
                break;
            }

        }

        // If best energy so far or fist iteration
        if(energy < bestEnergy || globalIt == 0)
        {
            bestEnergy = energy;
            for(int c = 0; c < K*numDims; ++c)
                bestCentroids[c] = centroids[c];
            for(int sample = 0; sample < numSamples; ++sample)
                bestLabels[sample] = labels[sample];
        }
    }

    for(int c = 0; c < K*numDims; ++c)
        centroids[c] = bestCentroids[c];
    for(int sample = 0; sample < numSamples; ++sample)
        labels[sample] = bestLabels[sample];
    // mexPrintf("\nEnd of K-Means\n");
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    /* Macros for the input arguments */
    #define features_IN prhs[0]
    #define ar_IN prhs[1]
    #define K_IN prhs[2]
    #define numIter_IN prhs[3]

    /* Macros for the output arguments */
    #define centroids_OUT plhs[0]
    #define labels_OUT plhs[1]

    /* Check correctness of input/output arguments */

    // Number of input/output arguments.
    if(nrhs < 4 || nrhs > 4)
        mexErrMsgTxt("Wrong number of input arguments.");
    else if(nlhs > 2)
        mexErrMsgTxt("Too many output arguments.");

    /* Get input data */
    int numSamples = (int)mxGetM(features_IN);
    int numDims = (int)mxGetN(features_IN);
    double* features = (double*)mxGetData(features_IN);
    double* ar = (double*)mxGetData(ar_IN);
    int K = (int)mxGetScalar(K_IN);
    int numIter = (int)mxGetScalar(numIter_IN);

    /* Create output data */
    centroids_OUT = mxCreateNumericMatrix(K, numDims, mxDOUBLE_CLASS, mxREAL);
    double* centroids = (double*)mxGetData(centroids_OUT);
    labels_OUT = mxCreateNumericMatrix(numSamples, 1, mxDOUBLE_CLASS, mxREAL);
    double* labels = (double*)mxGetData(labels_OUT);

    /* Call method: KMeans9D */
    kMeans(centroids, labels, numSamples, numDims, features, ar, K, numIter);

    return;
}

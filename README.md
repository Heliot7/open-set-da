# (Open Set) Domain Adaptation for image classification tasks
ICCV'17 paper at: http://pages.iai.uni-bonn.de/gall_juergen/download/jgall_opensetdomain_iccv17.pdf

-> Tested on Matlab 2013b - Windows 7
-> Caffe binaries compiled on Visual Studio 2013 / Matlab 2013b (please, use your own binaries or pre-computed features

Start the classification task:
- main.m

Modify parameters:
- InputParameters.m
  - isDA = true activates domain adaptation 
  - "ATI" is our developed method 
  - numSrcClusters must contain the same number as classes or viewpoints, so "ATI" works 
  - Saenko = 10, Office = 31, Viewpoints = 8, 16, 24, 36, ...
 
Datasets:
- For image classification: Saenko, Office and Sentiment datasets are standard evaluation datasets, select the same class in InputParameters.m ("sourceDataset" and "targetDataset") and then update accordingly "source" and "target" options within their classes to change the different domains. Better with CNN-fc7 features:
  - Download office dataset: https://people.eecs.berkeley.edu/~jhoffman/domainadapt/ and extract into Data\Real\DomainAdaptation\Saenko\  folder
  - Sentiment Dataset: 400-dim Bag of Words extraction in Data\Real\DomainAdaptation\Sentiment\ folder
- For viewpoint refinement/estimation: Synthetic data as "sourceDataset" and EPFL, ObjectCat3D, Pascal3D and Imagenet3D as "targetDataset" in InputParameters.m. Better with features that preserve layout information: CNN-pool5 or HOG.

Important files:
- step1_Classification.m
- DA_ATI.m
- computeCorrespondences.m

Results:
- files with (el) compute the mean accuracy among all test data elements/instances.
- files with (pr) compute the mean of all class mean accuracies.

for any question, please contact me: panareda@gmail.com, s6papana@uni-bonn.de

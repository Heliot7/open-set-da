% Class file with all input parameters.
classdef InputParameters < dynamicprops
    properties        
        %% Pipeline and samples %%
        % - Where the dataset folder hangs
        PATH_DATA = 'Z:\PhD\Data\';
        % - Storage of results
        PATH_RESULTS = 'Z:\PhD\Results\Test_github\';
        % Type of object recognition task
        typePipeline = 'class'; % ['class']
        % - Training per class (+Inf means that all images are included)
        numSrcTrain = +Inf;
        numTgtTrain = +Inf;
        numTgtTrainDA = +Inf;
        % - Output parameters
        isSaveTSNE = false;
        isStoreTransferOutput = true;
        isShowPose = false;
        % - Random seed for tests (selected images, candidates...)
        seedRand = 1; % Select -1 if you take ALL samples
        
        %% Datasets %%
        % - Name of dataset, based on Matlab clsases in "datasets" folder
        % Saenko, Office, Sentiment and other DA datasets contain source
        % and target sub-datasets
        % - Synthetic and Object datasets do DA for pose estimation
        % (specify number of viewpoints in class)
        sourceDataset = Office;
        targetDataset = Office;

        % - Feature descriptors
        typeDescriptor = 'CNN-fc7'; % ['HOG', 'BoW', 'CNN-fc7', 'CNN-pool5']
        % - Warp or keep Aspect Ratio when reading data
        keepAR = false;
        % - Apply ZScore to data
        isZScore = true;
        % - CNN 
        PATH_CNN = 'Z:\PhD\Data\CNN\';
        % - Name of folder with CNN weights and models
        cnnName = 'AlexNet'; % ['AlexNet', 'GoogleNet', 'VGG', 'ResNet']
        % - Name of *.caffemodel
        cnnModel = 'AlexNet'; % ['AlexNet'] 
        
        %% Classifiers %%
        typeClassifier = 'LSVM'; % ['LSVM', 'kNN', 'CNN']
        % - Precondition in supervision: 'class' attribute known
        isClassSupervised = true;
        is4ViewSupervised = false;
        % - LSVM
        methodSVM = 'libsvm' % 'libsvm' (LSVM & SVM), 'liblinear' (LSVM)
        multiClassApproach = 'OVO'; % 'OVA': One-vs-All (libsvm), 'OVO': One-vs-One (liblinear)
        C_LSVM = 0.001; % C param in LSVM
        CV_LSVM = false; % Cross Validation 
        %% Special DA Cases %%
        % - Openset Domain adaptation (some classes left unknown
        isOpenset = true;
        % - Apply LSVM with unknown classes
        isWSVM = false;
        % - extra type of data used for training with domain adaptation data
        trainDomain = 'tgt_gt'; % ['tgt_gt', 'tgt', 'both']
        %% Domain Adaptation %%
        isDA = true;
        typeDA = 'ATI'; % ['ATI', 'gfk', SA', 'MMDT', 'CORAL', ...}
        % 'whitening', 'gfk', 'MMDT', 'shekhar', 'saenko', 'DASVM'
        % - class-based problems
        daAllSrc = false; % true, takes all source samples, false only standard subset
        daAllTgt = true; % true, takes all target samples, false only standard subset
        daOnlySupervised = false; % true, labelled test data is embedded in training (label 0)
        daNumSupervised = 3; % Number of test tasmple with known labels
        % - FMO attributes
        iterResetDA = 1; % Start with old assignments but Id transf. matrix
        numIterATI = 1; % How many times we run ATI using previous results
        iterDA = 6; % Number of iteration of ATI method (<- MAIN ITERATION PARAM)
        numIterOpt = 33; % Number of iterations in optimisation process
        transformationDomain = 'src';
        dimPCA = 0.33; % PCA reduction
        numTgtClusters = 99999; % large enough to get 1 cluster = 1 sample
        numSrcClusters = 11; % Must be the same number as number of classes / viewpoints
        deltaW = 1.0; % 1.0 = all transformation applied, [0 < deltaW < 1] transforms slower and might converge better (e.g. = 0.5)
        numCorr = 99; % extra source for unbalanced datasets [1..Inf]*(numTgt/numSrc)
        % e.g. numCorr = 99 times more samples assigned one specific class = in practice, no constraints
        % e..g numCorr = 1 source classes assigned the same number of times. If datasets are balanced results improve        
        numLambda = 1.0; % distance samples empty nodes <-> tgt samples
        tol_residual = 0.001;
        tol_W = 0.0;
        %% LC
        numNN = 0; % locality constraint = 1
        isClosestNN = false;
        isFMOAllSamples = true;
        %% Background handle
        % - Include Bg samples in correspondence estimation
        includeBgClass = true;
        % - Ignore unknown classes
        isWild = false;
        % - Type of supervision (protocol Office or 'given # samples')
        typeWildSupervision = 'Office';
        % Print-outs
        isMidResultsDA = true; % Compute additional classifiers for in-between results
        isDaView2D = false;
        % Special test cases 
        daSpecial = ''; % {'' (standard), 'gt' (ground truth assignments, 'rnd' (random assignments)};
    end

end
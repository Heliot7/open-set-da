function Run_DA()
    % -> List of parameters
    listDatasets = {'Office','Saenko','Testbed','CrossDataset','Sentiment'};
    numSrcClusters = [31, 10, 40, -1, 2];
    listFeatures = {'CNN-fc7', 'BoW', 'CNN-fc7_FT'};
    listClasses = { ...
        {'AMAZON', 'DSLR', 'WEBCAM'}, ...
        {'AMAZON', 'DSLR', 'WEBCAM', 'CALTECH'}, ...
        {'CALTECH256', 'IMAGENET','SUN', 'BING'}, ...
        {'caltech101', 'office', 'pascal07'}, ... % {'CALTECH256', 'IMAGENET','SUN', 'BING'}, ... % {'caltech101', 'imagenet', 'msrcorid', 'bing', 'pascal07', 'eth80', 'office'}, ...
        {'books', 'dvd', 'elec', 'kitchen'}};
    isClassSupervised = [false, true];
    typeDA = {'','FMO','gfk','SA','CORAL','TCA', 'denoising', 'JGSA','corr'}; % Saenko? semi-sup?
    isDA = [false, true, true, true, true, true, true];
    typeCaffe = {'AlexNet', 'DA-BP', 'DA-DAN', 'DA-RTN', 'DA-Tzeng'};
    
    incr = [20 40 60 80 100 200 300 400];
    
    % SetUp of DA parameters
    da.numCorr =   [99,  99,  99,  99,  99];
    da.numLambda = [1.0, 0.5, 0.4, 0.5, 0.33];
    da.delta =     [0.3, 0.3, 1.0, 0.3, 0.3];
    da.numNN =     [0,   0,   1,   2,   1];
    da.PCA = [256, 512, 1000]; % GFK: [20,100, 256]; % TCA: [30, 100, 256]; % SA: [80, 256, 500];
    
    nameRun = 'incr_3\';
    for sup = 1 % length(isClassSupervised)
        for d = 1 %length(listDatasets)
            classes = listClasses{d};
            for feat = 1 %length(listFeatures)
                for m = 2 % length(typeDA)
%                     numTechniques = 1;
%                     if(m == 2) % FMO
%                         numTechniques = 8;
%                     end
                    for src = 1 % 1:length(classes)
                        for tgt = 2 % 2:length(classes)
                            for it = 1:5
                                for t = 1 % numTechniques)
                                    for c = 1:length(incr) % 1:length(typeCaffe);
                                        
                                        if(src == tgt)
                                            continue;
                                        end
                                        % PARAMETERS
                                        input = InputParameters;
                                        % Temporal params:
                                        input.isRebuttal = true;
                                        input.isOpenset = true;
                                        input.isWSVM = false;
                                        % Rebuttal ICCV 
                                        input.numImages = incr(c);
                                        
                                        input.isSaveTSNE = false;
                                        
                                        % -> Dataset specific cases
    %                                     if(strcmpi(listDatasets{d},'CrossDataset'))
    %                                         % if(protocolSparse(classes{src}, classes{tgt}))
    %                                             continue;
    %                                         end
    %                                     end
                                        if(strcmpi(listDatasets{d},'Sentiment'))
                                            if(~(strcmpi(classes{src},'kitchen') && strcmpi(classes{tgt},'dvd') || ...
                                                    strcmpi(classes{src},'dvd') && strcmpi(classes{tgt},'books') || ...
                                                    strcmpi(classes{src},'books') && strcmpi(classes{tgt},'elec') || ...
                                                    strcmpi(classes{src},'elec') && strcmpi(classes{tgt},'kitchen') ))
                                                continue;
                                            end
                                        end

                                        % -> Classifier
                                        input.PATH_DATA = 'Z:\PhD\Data\';
                                        if(input.isWSVM)
                                            input.methodSVM = 'libsvm-open';
                                            input.multiClassApproach = 'OVA';
                                        else
                                            input.methodSVM = 'libsvm';
                                            input.multiClassApproach = 'OVO';
                                        end
                                        input.C_LSVM = 0.001;
                                        input.CV_LSVM = false;
                                        % -> Save results
                                        input.PATH_RESULTS = ['Z:\ICCV\' nameRun num2str(input.numImages) '\' num2str(it) '\'];
                                        % -> Datasets
                                        input.sourceDataset = eval(listDatasets{d});
                                        input.sourceDataset.source = classes{src};
                                        input.targetDataset = eval(listDatasets{d});
                                        input.targetDataset.target = classes{tgt};
                                        % -> Protocol
                                        input.daAllSrc = true;
                                        input.daAllTgt = true;
                                        % -> Features
                                        input.typeDescriptor = listFeatures{feat};
                                        strOS = 'CS';
                                        if(input.isRebuttal && input.isOpenset)
                                            strOS = 'OS';
                                        elseif(input.isRebuttal && input.isWSVM)
                                            strOS = 'CSOS';
                                        elseif(input.isRebuttal)
                                            strOS = 'OSno';
                                        elseif(~input.isRebuttal && ~input.isOpenset && d == 2)
                                            strOS = 'SA';
                                        end
%                                         if(c == 1)
                                            input.cnnName = 'AlexNet';
                                            input.cnnModel = 'AlexNet';
%                                         elseif(c > 1 && c < 5)
%                                             input.cnnName = [typeCaffe{c} '-' strOS];
%                                             caffeSrcTgt = getOfficeData(src, tgt);
%                                             input.cnnModel = [typeCaffe{c} '-' strOS '-' caffeSrcTgt];
%                                         elseif(c == 5)
%                                             input.cnnName = [typeCaffe{c} '-' strOS];
%                                             caffeSrcTgt = getOfficeData(src, tgt);
%                                             input.cnnModel = [typeCaffe{c} '-' strOS '-' caffeSrcTgt '-r' num2str(it)];
%                                         end
                                        input.isZScore = true;
                                        % -> Domain Adaptation
                                        input.isDA = isDA(m);
                                        input.typeDA = typeDA{m};
                                        input.numTgtClusters = 99999;
                                        if(input.isOpenset)
                                            input.numSrcClusters = 11; % numSrcClusters(d);
                                        elseif(input.isWSVM)
                                            input.numSrcClusters = 10;                                            
                                        else
                                            if(~input.isRebuttal)
                                                input.numSrcClusters = numSrcClusters(d);
                                            else
                                                input.numSrcClusters = 10;
                                            end
                                        end
                                        input.iterResetDA = 1;
                                        input.iterDA = 5;
                                        input.numIterFMO = 1;
                                        input.numIterOpt = 50;
                                        if(m > 2)
                                            input.dimPCA = da.PCA(t);
                                        else
                                            input.dimPCA = 0.3;
                                        end
                                        input.tol_residual = 0.01;
                                        input.deltaW = da.delta(t);
                                        input.isClosestNN = false;
                                        input.numNN = da.numNN(t);
                                        input.numCorr = da.numCorr(t);
                                        input.numLambda = da.numLambda(t);
                                        if(da.numLambda(t) > 1)
                                            input.dynLambda = true;
                                        else
                                            input.dynLambda = false;
                                        end
                                        input.initDynLambda = 0.8;
                                        input.limDynLambda = 0.4;
                                        input.stepDynLambda = 0.1;
                                        input.daSpecial = '';
                                        % -> Open Set
                                        input.includeBgClass = true;
                                        input.isWild = false;
                                        input.typeWildSupervision = 'Office';
                                        % -> Supervision
                                        input.isClassSupervised = isClassSupervised(sup);                        
                                        input.daOnlySupervised = false;
                                        input.daNumSupervised = 3;
                                        % -> RUN! 
                                        input.isMidResultsDA = true;
                                        input.seedRand = it;
                                        if(it == 6)
                                            input.seedRand = 1;
                                        end
                                        main(input);
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end
    
function isContinue = protocolSparse(src, tgt)
    isContinue = true;
    if(strcmpi(src,'caltech256') && strcmpi(tgt,'imagenet') || strcmpi(src,'caltech256') && strcmpi(tgt,'sun') || ...
            strcmpi(src,'imagenet') && strcmpi(tgt,'caltech256') || strcmpi(src,'imagenet') && strcmpi(tgt,'sun') || ...
            strcmpi(src,'sun') && strcmpi(tgt,'caltech256') || strcmpi(src,'sun') && strcmpi(tgt,'imagenet') || ...
            strcmpi(src,'eth80') && strcmpi(tgt,'caltech256') || strcmpi(src,'eth80') && strcmpi(tgt,'pascal07') || ...
            strcmpi(src,'office') && strcmpi(tgt,'caltech256') || strcmpi(src,'office') && strcmpi(tgt,'pascal07') || ...
            strcmpi(src,'bing') && strcmpi(tgt,'caltech256') || strcmpi(src,'bing') && strcmpi(tgt,'pascal07'))
        isContinue = false;
    end
end

function isContinue = protocolSparse2(src, tgt)
    isContinue = true;
    if(strcmpi(src,'caltech101') && strcmpi(tgt,'imagenet') || strcmpi(src,'caltech101') && strcmpi(tgt,'sun') || ...
            strcmpi(src,'imagenet') && strcmpi(tgt,'caltech101') || strcmpi(src,'imagenet') && strcmpi(tgt,'sun') || ...
            strcmpi(src,'sun') && strcmpi(tgt,'caltech101') || strcmpi(src,'sun') && strcmpi(tgt,'imagenet') || ...
            strcmpi(src,'eth80') && strcmpi(tgt,'caltech101') || strcmpi(src,'eth80') && strcmpi(tgt,'pascal07') || ...
            strcmpi(src,'office') && strcmpi(tgt,'caltech101') || strcmpi(src,'office') && strcmpi(tgt,'pascal07') || ...
            strcmpi(src,'pascal07') && strcmpi(tgt,'caltech101') || strcmpi(src,'caltech101') && strcmpi(tgt,'pascal07'))
        isContinue = false;
    end
end

function str = getOfficeData(src, tgt)
    
    str = '';
    switch(src)
        case 1
            str = 'A'; % Amazon
        case 2
            str = 'D'; % DSLR            
        case 3
            str = 'W'; % Webcam
        case 4
            str = 'C'; % Caltech
    end
    
    switch(tgt)
        case 1
            str = [str 'A']; % Amazon
        case 2
            str = [str 'D']; % DSLR
        case 3
            str = [str 'W']; % Webcam
        case 4
            str = [str 'C']; % Caltech
    end

end

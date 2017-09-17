function [data, features, testData, testFeatures] = getOfficePrecomputed_CNN(input, dataset_info, phase)

    features = []; testFeatures = [];
    data.annotations.classes = []; testData.annotations.classes = [];
    dataset = lower(dataset_info.(phase));
    if(strcmpi(phase,'source'))
        if(input.daAllSrc)
            numImgs = 99999;
        else
            numImgs = 20;
            if(strcmpi(dataset,'WEBCAM') || strcmpi(dataset,'DSLR'))
                numImgs = 8;
            end
        end
    elseif(strcmpi(phase,'target'))
        if(input.isClassSupervised)
            numImgs = 3;
        else % unsupervised
            numImgs = 99999;
        end
    end

    % NOTE: Change everytime we run new experiments to get new results
    rng(input.seedRand); 
    
    % -> Pre-processing (saving all images in one file to speed-up
%     path = [input.PATH_DATA dataset_info.path 'CNN-pre\' dataset '\decaf-fts\'];
%     folder = dir(path);
%     allClassFolders = {folder.name};
%     [~, ~, exts] = cellfun(@fileparts, allClassFolders, 'UniformOutput', false);
%     idxPaths = cellfun(@(x) sum(x), cellfun(@(x) strcmp(x, {''}), exts, 'UniformOutput', false)) == 1;
%     allClassFolders = allClassFolders(idxPaths)';
%     features = [];
%     labels = [];
%     for i = 1:length(allClassFolders)
%         
%         pathClass = [path allClassFolders{i} '\'];
%         folder = dir(pathClass);
%         allMat = {folder.name};
%         [~, ~, exts] = cellfun(@fileparts, allMat, 'UniformOutput', false);
%         idxPaths = cellfun(@(x) sum(x), cellfun(@(x) strcmp(x, {'.mat'}), exts, 'UniformOutput', false)) == 1;
%         allMat = allMat(idxPaths)';
%         for j = 1:length(allMat)
%             load([pathClass allMat{j}]);
%             features = [features; fc7'];
%         end
%         % labels = [labels; repmat(allClassFolders(i),length(allMat),1)];
%         labels = [labels; dataset_info.classes(i*ones(length(allMat),1))'];
%         
%     end
%     save([input.PATH_DATA dataset_info.path 'CNN-pre\' dataset '-fc7'], 'features', 'labels', '-v7.3');
%     keyboard;
    
    load([input.PATH_DATA dataset_info.path 'CNN-pre\' dataset '-fc7']);
    if(input.isZScore)
        features = features./ repmat(sum(features,2),1,size(features,2));
        features = zscore(features);
    end
    rawFeat = features;
        
    features = [];
    for i = 1:31 % numClasses = 31
        classSamples = rawFeat(ismember(labels,dataset_info.classes{i}),:);
        permSamples = randperm(size(classSamples,1));
        % Train portion
        trainSamples = permSamples(1:min(length(permSamples), numImgs));
        features = [features; classSamples(trainSamples,:)];
        data.annotations.classes = [data.annotations.classes; dataset_info.classes(i*ones(length(trainSamples),1))'];
        % Test portion
        if(numImgs < size(classSamples,1) && strcmpi(phase,'target'))
            testSamples = classSamples(permSamples(numImgs+1:end),:);
            testFeatures = [testFeatures; testSamples];
            testData.annotations.classes = [testData.annotations.classes; dataset_info.classes(i*ones(size(testSamples,1),1))'];
        end
    end
    
    if(isempty(testData.annotations.classes))
        testData = [];
    end
    
end

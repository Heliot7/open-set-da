function main(input)

    clc;
    close all;
    warning ('off','all');
    global netCaffe;
    
    if(nargin == 0)
        input = InputParameters;
    end
    
    % Load CNN model if specified
    % > Comment "if" statement if features/CNN/caffe_.mexw64 not compatible
    if(~isempty(strfind(input.typeDescriptor,'CNN')))
        desc = input.typeDescriptor(5:end);
        model = [input.PATH_CNN input.cnnName '\' input.cnnName '-deploy-' desc '.prototxt'];
        weights = [input.PATH_CNN input.cnnName '\' input.cnnModel '.caffemodel'];
        caffe.set_mode_gpu();
        caffe.set_device(0);
        netCaffe = caffe.Net(model, weights, 'test');
        fprintf('==========\n');
        fprintf('CNN Net: %s\nWeights: %s\nOutput layer: %s\n', input.cnnName, input.cnnModel, desc);
        fprintf('==========\n');
    end
    
    if(strcmpi(input.typePipeline,'class'))
        fprintf('\nMulti-Object/Viewpoint Classification:\n');
    elseif(strcmpi(input.typePipeline,'det'))
        fprintf('\nObject Detection:\n');
    elseif(strcmpi(input.typePipeline,'classDet'))
        fprintf('\n2-Step Object Detection and Meta-Data estimation:\n');
    end
    fprintf('Class(es): ');
    for i = 1:length(input.sourceDataset.classes)
        fprintf('%s ', input.sourceDataset.classes{i});
    end
    fprintf('\n');

    % Prepare discrete list of azimuth viewpoint angles (if not regression)
    if(isprop(input.sourceDataset, 'azimuth') && ~isempty(input.sourceDataset.azimuth))
        input.sourceDataset.azimuth = getDiscreteAzimuths(input.sourceDataset.azimuth, input.is4ViewSupervised);
    else
        input.is4ViewSupervised = false; % No viewpoints to coarse them
    end
    if(isprop(input.sourceDataset,'azimuth') && isprop(input.targetDataset, 'azimuth')); % && ~isempty(input.targetDataset.azimuth))
        input.targetDataset.azimuth = input.sourceDataset.azimuth;
        if(input.is4ViewSupervised) % Cannot be 4view:true sup:false!
            input.isClassSupervised = true;
        end
    else
        input.is4ViewSupervised = false; % No viewpoints to coarse them
    end
    
    % -> Get source dataset (train)
    fprintf('Source dataset: %s', class(input.sourceDataset));
    if(isprop(input.sourceDataset,'source'))
        fprintf(' [%s] ', input.sourceDataset.source);
    end
    fprintf('\n');
    [input, srcData, srcFeatures] = getData(input, 'source');
    % -> Set Up Bounding boxes and features for each class & view
    srcData = setupData(input.sourceDataset, srcData);    
    
    % -> Get target dataset (train + test)
    fprintf('Target dataset: %s', class(input.targetDataset));
    if(isprop(input.targetDataset,'target'))
        fprintf(' [%s] ', input.targetDataset.target);
    end
    fprintf('\n');
    [input, tgtData, tgtFeatures, testData, testFeatures] = getData(input, 'target');
    fprintf('\n');

    if(input.isOpenset && (strcmpi(class(input.sourceDataset),'CrossDataset') || strcmpi(class(input.sourceDataset),'Video') || ...
            strcmpi(class(input.sourceDataset),'Visda17')))
        [input, srcData, srcFeatures, tgtData, tgtFeatures, testData, testFeatures] = ...
            intersectionCrossDataset(input, srcData, srcFeatures, tgtData, tgtFeatures);
        input.numSrcClusters = length(input.sourceDataset.classes);
        if(strcmpi(class(input.sourceDataset),'Visda17'))
            % Sort samples in order of getFeature loading
            % - Source data
            auxClasses = [];
            auxPaths = [];
            for i = 1:length(input.sourceDataset.classes)
                isClass = ismember(srcData.annotations.classes,input.sourceDataset.classes(i));
                auxClasses = [auxClasses; srcData.annotations.classes(isClass)];
                auxPaths = [auxPaths; srcData.imgPaths(isClass)];
            end
            srcData.annotations.classes = auxClasses;
            srcData.imgPaths = auxPaths;
        end
    end
    
    % Check error in compatible viewpoints    
    if(strcmpi(input.typePipeline, 'det') && strcmpi(input.trainDomain,'both') && ...
            (~isprop(input.sourceDataset,'azimuth') && isprop(input.targetDataset,'azimuth')) && ...
            (isprop(input.sourceDataset,'azimuth') && ~isprop(input.targetDataset,'azimuth')))
        error('[[Caught ERROR: One domain lacks viewpoints: %s]]', phase, class(dataset));
    elseif(strcmpi(input.typePipeline, 'det') && strcmpi(input.trainDomain,'both') && ...
            isprop(input.sourceDataset,'azimuth') && isprop(input.targetDataset,'azimuth') && ...
            length(input.sourceDataset.azimuth) ~= length(input.targetDataset.azimuth))
        error('[[Caught ERROR: Domains have different viewpoint granularity: %s]]', phase, class(dataset));
    end
    
    % -> Prepare output folders
    mDir = [getResultsPath(input) '/'];
    removeDir(mDir);
    createDir(mDir);

    % -> Gathering data features
    input = setupFeatures(input, srcData);
    % Load source 
    fprintf('Loading source features\n');
    if(isempty(srcFeatures) && isfield(srcData,'annotations'))
        srcFeatures = getFeatures(input, input.sourceDataset, srcData, input.numSrcTrain);
        srcData.annotations.imgId  = srcData.annotations.imgId(1:size(srcFeatures));
        if(isfield(srcData.annotations,'BB'))
            srcData.annotations.BB = srcData.annotations.BB(1:size(srcFeatures),:);
        end
        srcData.annotations.classes = srcData.annotations.classes(1:size(srcFeatures));
        if(isfield(srcData.annotations,'vp'))
            srcData.annotations.vp.azimuth  = srcData.annotations.vp.azimuth(1:size(srcFeatures));
        end
    end
    
    % Load target features (train)
    fprintf('Loading target features (train)\n');
    if(isempty(tgtFeatures) && isfield(tgtData,'annotations'))
        tgtFeatures = getFeatures(input, input.targetDataset, tgtData, input.numTgtTrain);
    end
    fprintf('Loading target features (test)\n');
    if(isempty(testFeatures) && isfield(testData,'annotations'))
        testFeatures = getFeatures(input, input.targetDataset, testData, +Inf, '_test');
    end
    
    % -> Classification (Label Transfer)
    vpClassifiers = [];
    [~, ~, metadata] = getIdLabels(input.sourceDataset, srcData.annotations);
    if(strfind(lower(input.typePipeline),'class'))  
        [transferLabels, srcFeatures, vpClassifiers] = step_Classification(input, srcData, srcFeatures, tgtData, tgtFeatures, testData, testFeatures);
        if(strfind(lower(input.typePipeline),'det'))
            for i = 1:length(metadata)
                values = transferLabels(:,i);
                if(strcmpi(metadata{i},'azimuth'))
                    values = cell2mat(cellfun(@str2num, values, 'UniformOutput', false));
                    input.targetDataset.azimuth = input.sourceDataset.azimuth;
                end
                tgtData.annotations.(metadata{i}) = values;
            end
        end
    end

    % -> Step 2: Object Detection + Viewpoint/Meta-data Estimation    
    if(strfind(lower(input.typePipeline),'det'))
        % Load training data
        trainData = [];
        fprintf('Loading training features\n');
        if(strcmpi(input.trainDomain,'src'))
            trainData = srcData;
            trainFeatures = srcFeatures;
            trainInfo = input.sourceDataset;
        elseif(strcmpi(input.trainDomain,'tgt'))
            if(strcmpi(input.sourceDataset, input.targetDataset))
                trainData = srcData;
                trainFeatures = srcFeatures;
                trainInfo = input.sourceDataset;
            else
                trainData = tgtData;
                trainFeatures = tgtFeatures;
                trainInfo = input.targetDataset;
            end
        elseif(strcmpi(input.trainDomain,'tgt_gt'))
            trainData = tgtData;
            trainFeatures = tgtFeatures;
            trainInfo = input.targetDataset;
        elseif(strcmpi(input.trainDomain,'both'))
            trainData.imgPaths = [srcData.imgPaths; tgtData.imgPaths];
            trainData.annotations.imgId = [srcData.annotations.imgId; tgtData.annotations.imgId + length(srcData.imgPaths)];
            trainData.annotations.BB = [srcData.annotations.BB; tgtData.annotations.BB];
            [~, ~, tgtMetadata] = getIdLabels(input.targetDataset, tgtData.annotations, metadata);
            for i = 1:length(tgtMetadata)
                trainData.annotations.(tgtMetadata{i}) = [srcData.annotations.(tgtMetadata{i}); tgtData.annotations.(tgtMetadata{i})];
            end
            trainInfo = input.sourceDataset;
            trainInfo.classes = unique([trainInfo.classes, input.targetDataset.classes]);
            trainFeatures = [srcFeatures; tgtFeatures];
        end
        trainData = setupData(trainInfo, trainData);
        step2_ObjectDetection(input, trainInfo, trainData, trainFeatures, testData, vpClassifiers);
    end
    
    if(~isempty(strfind(input.typeDescriptor,'CNN')))
        clearvars -global netCaffe;
        caffe.reset_all();
    end
    
end


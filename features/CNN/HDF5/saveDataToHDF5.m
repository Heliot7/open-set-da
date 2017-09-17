function numClasses = saveDataToHDF5(input, sampleSize, numNegatives)

    savePath = [input.PATH_DATA input.sourceDataset.path 'samples\'];

    % Get training dataset
    [input, data] = getData(input, 'source');
    data_info = input.sourceDataset;
    data = setupData(data_info, data);    
    
    % Get training images with desired shape 
    if(isprop(input.sourceDataset, 'viewpoints') && input.sourceDataset.viewpoints > 0)
        data_info.viewpoints = getDiscreteAzimuths(data_info.viewpoints, input.is4ViewSupervised);
    end
    samples = getImgSamples(input, data_info, data, sampleSize);
    
    % Get labels per training dataset
    [srcIds, convertIds] = getIdLabels(data_info, data.annotations);
    ids = zeros(1,size(srcIds,1));
    for i = 1:size(convertIds,1)
        matches = sum(ismember(srcIds, convertIds(i,:)),2) == size(convertIds,2);
        ids(matches) = i-1;
    end

    % In case there is only 1 class... add negative images
    if(max(ids) == 0)
        samples = cat(4, samples, getImgNegativeSamples(input, data, sampleSize, numNegatives));
        ids = [ids, ones(1, numNegatives)];
        numClasses = 2;
    else
        numClasses = max(ids+1);
    end
    
    % Pre-processing samples
    % - RGB -> BGR
    samples = samples(:, :, [3, 2, 1], :);
	% - Col(Width)*Row(Height)*Ch*Number
    samples = permute(samples,[2 1 3 4]);
    % - Mean substraction
    mean_data = caffe.io.read_mean([strrep(input.PATH_CNN, '\', '/') 'AlexNet/AlexNet-mean.binaryproto']);
    mean_channels = mean(mean(mean_data));
    % normal = (max([255; 255; 255] - mean_channels(:), mean_channels(:)))*2;
    % Shuffle samples + ids
    perms = randperm(length(ids));
    
    ids = ids(perms);

    % Save HDF5 database
    if(~exist(savePath, 'dir'))
        mkdir(savePath);
    end
    
    idxChunk = 1;
    sizeChunk = 5000;
    trainPaths = [];
    valPaths = [];
    idx = 1;
    while idxChunk <= size(samples,4)
        
        numChunks = min(sizeChunk,size(samples,4)-idxChunk+1);

        samples_chunk = single(samples(:,:,:,perms(idxChunk:idxChunk+numChunks-1)));
        ids_chunk = single(ids(idxChunk:idxChunk+numChunks-1));
        % Normalise data
        parfor i = 1:3
            samples_chunk(:,:,i,:) = (samples_chunk(:,:,i,:) - mean_channels(i)); %  ./ ...
                % repmat(std(samples_chunk(:,:,i,:),[],4), [1 1 1 numChunks]);
        end        
        samples_chunk = zscore(samples_chunk);
        
        % Separate between train and test data
        numTrain = floor(0.8*numChunks);
        numVal = numChunks - numTrain;
        
        % -> Train part
        saveTrain = [savePath 'data_train_' num2str(idx) '_' num2str(idxChunk) '.h5'];
        h5create(saveTrain, '/data', [sampleSize(1), sampleSize(2), 3, numTrain],'Datatype','single');
        h5create(saveTrain, '/label', numTrain,'Datatype','single');
        h5write(saveTrain, '/data', samples_chunk(:,:,:,1:numTrain));
        h5write(saveTrain, '/label', ids_chunk(1:numTrain));
        trainPaths = [trainPaths; {saveTrain}];
        
        % -> Validation part
        saveVal = [savePath 'data_val_' num2str(idx) '_' num2str(idxChunk) '.h5'];
        h5create(saveVal, '/data', [sampleSize(1), sampleSize(2), 3, numVal],'Datatype','single');
        h5create(saveVal, '/label',  numVal,'Datatype','single');
        h5write(saveVal, '/data', samples_chunk(:,:,:,numTrain+1:end));
        h5write(saveVal, '/label', ids_chunk(numTrain+1:end));
        valPaths = [valPaths; {saveVal}];

        idxChunk = idxChunk + sizeChunk;
        idx = idx + 1;

    end
   
    % Creat txt files with paths
    fid = fopen([strrep(savePath,'\','/') 'train.txt'], 'w');
    for i = 1:length(trainPaths)
        fprintf(fid,strrep(trainPaths{i},'\','/'));
        fprintf(fid,'\n');
    end
    fclose(fid);
    fid = fopen([strrep(savePath,'\','/') 'val.txt'], 'w');
    for i = 1:length(valPaths)        
        fprintf(fid,strrep(valPaths{i},'\','/'));
        fprintf(fid,'\n');
    end
    fclose(fid);
    
end



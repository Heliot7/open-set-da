function saveDataToImg(input, sampleSize, numNegatives)

    savePath = [input.PATH_DATA input.sourceDataset.path 'img_samples\'];

    % Get training dataset
    [input, data] = getData(input, 'source');
    data_info = input.sourceDataset;
    data = setupData(data_info, data);    
    
    % Get training images with desired shape
    if(isprop(input.sourceDataset, 'azimuth') && ~isempty(input.sourceDataset.azimuth) && input.sourceDataset.azimuth > 0)
        data_info.azimuth = getDiscreteAzimuths(data_info.azimuth, input.is4ViewSupervised);
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
    if(numNegatives > 0)
        samples = cat(4, samples, getImgNegativeSamples(input, data, sampleSize, numNegatives));
        ids = [ids, (max(ids)+1)*ones(1, numNegatives)];
    end
   
    % Permutation ids
    permIds = randperm(length(ids));
    ids = ids(permIds);
    samples = samples(:,:,:, permIds);
    
    idsTrain = ids(1:round(length(ids)*0.8));
    idsVal = ids(length(idsTrain)+1:end);
    
    fprintf('Storing image patches and ids\n');
    createDir(savePath);
	saveImgs(savePath, sampleSize, samples);
    saveFilePaths(savePath, idsTrain, idsVal, false);
    
end

function saveImgs(mDir, sampleSize, listImgs)

    numSamples = size(listImgs,4);
    for idxImage = 1:numSamples
        img = uint8(reshape(listImgs(:,:,:,idxImage), [sampleSize(1) sampleSize(2) 3]));
        imwrite(img, [mDir 'img' sprintf('%05d', idxImage) '.png']);
        if(mod(idxImage, 1000) == 0)
            fprintf('%d/%d stored image patches\n', idxImage, numSamples);
        end
    end

end

% 'tmp_samples2\tmp_train.txt', 'tmp_samples2\tmp_test.txt
function saveFilePaths(mDir, trainLabels, testLabels, isAbsolutePath)

    fileTrain = fopen([mDir 'train.txt'], 'w');
    fileTest = fopen([mDir 'val.txt'], 'w');
    if(exist('isAbsolutePath','var') && ~isAbsolutePath)
        mDir = [];
    end
    for i = 1:length(trainLabels)
        fprintf(fileTrain, '%s %d\n', strrep([mDir 'img' sprintf('%05d', i) '.png'], '\', '/'), trainLabels(i));
    end
    fclose(fileTrain);
    for j = 1:length(testLabels)
        fprintf(fileTest, '%s %d\n', strrep([mDir 'img' sprintf('%05d', i+j) '.png'], '\', '/'), testLabels(j));
    end
    fclose(fileTest);
end


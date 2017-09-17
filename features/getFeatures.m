function allFeatures = getFeatures(input, dataset_info, data, numAnnotations, extraStrName)

    global netCaffe;

    allFeatures = [];
    if(nargin < 5)
        extraStrName = '';
    end

    % Refine feature descriptor's name
    descriptor = input.typeDescriptor;
    if(strfind(input.typeDescriptor, 'CNN'))
        descriptor = [input.cnnModel '-' input.typeDescriptor(end-4:end)];
    end

    for c = 1:length(dataset_info.classes)
        
        strRow = num2str(input.featureInfo.patchSize(1));
        strCol = num2str(input.featureInfo.patchSize(2));
        strAR = '_AR';
        if(~input.keepAR)
            strAR = '_noAR';
        end
        strOcc = '';
        if(isprop(dataset_info,'isOcclusions') && dataset_info.isOcclusions)
            strOcc = '_occ';
        end
        strMirror = '';
        if(input.isMirrored)
            strMirror = '_flip';
        end
        strDA = '';
        if(strcmpi(class(input.sourceDataset),'office') || strcmpi(class(input.sourceDataset),'saenko') || ...
                strcmpi(class(input.sourceDataset),'testbed') || strcmpi(class(input.sourceDataset),'crossdataset') || ...
                strcmpi(class(input.sourceDataset),'visda17'))
            strSup = 'uns';
            if(input.isClassSupervised)
                strSup = 'sup';
            end
            strPhase = '_target';
            strClass = lower(input.targetDataset.target);
            if(strcmpi(extraStrName,'_test'))
                strPhase = '';
            elseif(strfind(lower(data.imgPaths{1}), lower(input.sourceDataset.source)))
                strPhase = '_source';
                strClass = lower(input.sourceDataset.source);
            end            
            strOpenSet = '';
            if(strcmpi(dataset_info.classes{c}, 'zz_unknown'))
                strOpenSet = [strOpenSet '_' num2str(input.numImages)];
            end
            strDA = [strPhase '_' strClass(1:3) '_rnd' mat2str(input.seedRand) '_' strSup strOpenSet];
        end
        if(strcmpi(class(dataset_info),'Pascal3D'))
            if(dataset_info.addImageNet3D)
                extraStrName = [extraStrName '_I3D'];
            end
        end
        if(~strcmpi(class(dataset_info),'Synthetic'))
            matPath = [input.PATH_DATA dataset_info.path 'mat_features\' ...
                dataset_info.classes{c} '_' descriptor '_' strRow '-' strCol strOcc strAR extraStrName strMirror strDA];
        else % Our Synthetic is a special case
            matPath = [input.PATH_DATA dataset_info.path dataset_info.classes{c} '\' dataset_info.sub_path 'mat_features\' ...
                dataset_info.classes{c} '_' descriptor '_' strRow '-' strCol strAR extraStrName strMirror];
        end
        if(exist([matPath '.mat'], 'file'))

            load([matPath '.mat'], 'features');
            % Check whether the previous saved data is consistent
            if(input.featureInfo.featureDim ~= size(features,2))
                fprintf('Loaded features have wrong dimensionality.\n');
                fprintf('Positive training samples need to be recomputed.\n');
                error('Manually DELETE features and RESTART again!\n');
            end

        else

            if(~isempty(strfind(input.typeDescriptor,'CNN')))
                num_batch = 16;
                idxBatch = 1;
                % CNN features need to swap row-col's
                listImgs = zeros(input.featureInfo.patchSize(2), input.featureInfo.patchSize(1), 3, num_batch , 'single');
                netCaffe.blobs('data').reshape([input.featureInfo.patchSize(2), input.featureInfo.patchSize(1), 3, num_batch]);
                netCaffe.reshape();
            end

            % Count number of dataset samples
            isClass = ismember(data.annotations.classes, dataset_info.classes{c});
            classImgId = data.annotations.imgId(isClass);
            if(isfield(data.annotations,'BB'))
                classBB = data.annotations.BB(isClass,:);
            else
                classBB = [];
            end
            numSamples = length(classImgId);
            features = zeros(numSamples, input.featureInfo.featureDim, 'single');
            previousId = 0;
            for idxSample = 1:numSamples

                currentId = classImgId(idxSample);
                if(previousId ~= currentId)
                    img = imread(data.imgPaths{currentId});
                    previousId = currentId;
                end

                if(~isempty(classBB))
                    BB = num2cell(classBB(idxSample,:));
                    [row, col, height, width] = BB{:};
                else
                    row = 1; col = 1; height = size(img,1); width = size(img,2);
                end

                % Compute HOGs for current Sample (colour img considered)
                if(strcmp(input.typeDescriptor, 'HOG'))
                    border = 8;
                    if(input.keepAR)
                        init_patch = cropBB(img, row, col, double(height), double(width), input.featureInfo.patchSize, border);
                    else
                        % Not accurate (revise borders
                        b = 1;
                        init_patch = img(max(1,row-b):min(row+height-1+b,size(img,1)),max(1,col-b):min(col+width-1+b,size(img,2)),:);
                        sampleAR = size(init_patch,1) / size(init_patch,2);
                        patchAR = [size(init_patch,1) size(init_patch,2)] ./ input.featureInfo.patchSize+2;
                        if(sampleAR > 1.0)
                            width = width + 3*border/sampleAR*patchAR(2);
                            col = col - 1.5*border/sampleAR*patchAR(2);
                            bHeight = 1.5*border*patchAR(1);
                            bWidth = 1.5*border*patchAR(2);
                        else
                            height = height + 3*border*sampleAR*patchAR(1);
                            row = row - 1.5*border*sampleAR*patchAR(1);
                            bWidth = 1.5*border*patchAR(2);
                            bHeight = 1.5*border*patchAR(1);
                        end
                        init_patch = img(max(1,row-bHeight):min(row+height-1+bHeight,size(img,1)),max(1,col-bWidth):min(col+width-1+bWidth,size(img,2)),:);
                        topOffset = uint16(1-row-bHeight); bottomOffset = uint16(row+height-1+bHeight - size(img,1));
                        leftOffset = uint16(1-col-bWidth); rightOffset = uint16(col+width-1+bWidth - size(img,2));
                        init_patch = padarray(init_patch, double([topOffset leftOffset]), 'replicate', 'pre');
                        init_patch = padarray(init_patch, double([bottomOffset rightOffset]), 'replicate', 'post');
                        init_patch = imresize(init_patch, input.featureInfo.patchSize+2, 'bilinear');
                    end
                    if(input.isMirrored && mod(idxSample,2) == 0)
                        init_patch = flipdim(init_patch, 2);
                    end
                    patch = grey2rgb(init_patch);
                    f = mexFeatures(single(patch), border);
                    f = f(:,:,1:31); % Remove last bin dimension with 0s
                    features(idxSample,:) = f(:);
                elseif(~isempty(strfind(input.typeDescriptor,'CNN')))
                    border = 1;
                    if(input.keepAR)
                        init_patch = cropBB(img, row, col, double(height), double(width), input.featureInfo.patchSize, border, input.isScaleInvariance);
                        init_patch = init_patch(2:end-1,2:end-1,:);
                    else
                        init_patch = img(max(1,row-border):min(row+height-1+border,size(img,1)),max(1,col-border):min(col+width-1+border,size(img,2)),:);
                        topOffset = uint16(1-row-border); bottomOffset = uint16(row+height-1+border - size(img,1));
                        leftOffset = uint16(1-col-border); rightOffset = uint16(col+width-1+border - size(img,2));
                        init_patch = padarray(init_patch, double([topOffset leftOffset]), 'replicate', 'pre');
                        init_patch = padarray(init_patch, double([bottomOffset rightOffset]), 'replicate', 'post');
                        init_patch = imresize(init_patch, input.featureInfo.patchSize, 'bilinear');
                    end
                    if(input.isMirrored && mod(idxSample,2) == 0)
                        init_patch = flipdim(init_patch, 2);
                    end
                    patch = grey2rgb(init_patch);
                    patch = patch(:, : , [3, 2, 1]);
                    patch = permute(patch, [2, 1, 3]);
                    imgB = patch(:,:,1); imgG = patch(:,:,2); imgR = patch(:,:,3);
                    means = [mean(imgB(:)),mean(imgG(:)),mean(imgR(:))];
                    patch = single(patch);
                    patch = bsxfun(@minus, patch, reshape(means,1,1,3));
                    listImgs(:,:,:,idxBatch) = patch;
                    idxBatch = idxBatch + 1;
                    if(idxBatch > num_batch || idxSample >= numSamples)
                        % feat = CNN_Caffe('getFeatures', listImgs(:,:,:,1:idxBatch-1), input.typeDescriptor(5:end));
                        if(idxBatch <= num_batch)
                            netCaffe.blobs('data').reshape([input.featureInfo.patchSize(2), input.featureInfo.patchSize(1), 3, idxBatch-1]);
                            netCaffe.reshape();
                        end
                        res = netCaffe.forward({listImgs(:,:,:,1:idxBatch-1)});
                        feat = res{1}; % outputs last layer
                        for idxB = 1:idxBatch-1
                            len_array = length(size(feat));
                            if(len_array == 4)
                                f = feat(:,:,:,idxB);
                            elseif(len_array == 2)
                                f = feat(:,idxB);
                            end
                            f = permute(f, [2, 1, 3]);
                            features(idxSample-(idxBatch-1)+idxB,:) = f(:);
                        end
                        idxBatch = 1;
                    end
                end
                fprintf('Sample %d/%d CHECKED\n', idxSample, numSamples);            
                
                % Commented: - Test correctness
%                 figure(1);
%                 subplot(1,2,1);
%                 imshow(init_patch);
%                 axis off; axis image;
%                 subplot(1,2,2);
%                 imshow(img);
%                 rectangle('position', [col row width height], 'LineWidth', 1, 'EdgeColor', [0 0 1]);
%                 % pause(0.2);
%                 keyboard;

            end

            % features = zscore(features);
            fprintf('Saving pos matrix... ');
            posFolder = strfind(matPath, '\');
            createDir(matPath(1:posFolder(end)));
            save([matPath '.mat'], 'features', '-v7.3');
            fprintf('done!\n');

        end

        % Normalise as pre-processing step for BoW features
        % - (do not do it with CrossDataset)
%         if(strcmpi(input.typeDescriptor, 'BoW') && ~strcmpi(class(dataset_info),'CrossDataset_dense'))
%             features = zscore(features);
%         end
        
%         if(numAnnotations ~= +Inf)
%             if(input.is4ViewSupervised)
%                 data.annotations
%             else
                features = features(1:min(size(features,1),numAnnotations),:);
%             end
%         end
        allFeatures = [allFeatures; features];
        clear features;

    end

    if(input.isZScore)
        allFeatures = allFeatures./ repmat(sum(allFeatures,2),1,size(allFeatures,2));
        allFeatures = zscore(allFeatures);
    end
end


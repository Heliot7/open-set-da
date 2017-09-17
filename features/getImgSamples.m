function samples = getImgSamples(input, dataset_info, data, sampleSize)

    numGlobalSamples = length(data.annotations.imgId);
    samples = zeros(sampleSize(1),sampleSize(2),3,numGlobalSamples,'uint8');
    idxGlobalSample = 1;
    for c = 1:length(dataset_info.classes)

        % Count number of dataset samples
        isClass = ismember(data.annotations.classes, dataset_info.classes{c});
        classImgId = data.annotations.imgId(isClass);
        if(isfield(data.annotations,'BB'))
            classBB = data.annotations.BB(isClass,:);
        else
            classBB = [];
        end
        numSamples = length(classImgId);
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
            border = 1;
            if (input.keepAR)
                patch = cropBB(img, row, col, double(height), double(width), sampleSize, border);
                init_patch = patch(2:end-1,2:end-1,:);
            else % NO AR!!
                init_patch = img(max(1,row-border):min(row+height-1+border,size(img,1)),max(1,col-border):min(col+width-1+border,size(img,2)),:);
%             topOffset = uint16(1-row-border); bottomOffset = uint16(row+height-1+border - size(img,1));
%             leftOffset = uint16(1-col-border); rightOffset = uint16(col+width-1+border - size(img,2));
%             init_patch = padarray(init_patch, double([topOffset leftOffset]), 'replicate', 'pre');
%             init_patch = padarray(init_patch, double([bottomOffset rightOffset]), 'replicate', 'post');
            end
            patch = imresize(init_patch, sampleSize, 'bilinear');
            samples(:,:,:,idxGlobalSample) = grey2rgb(patch);
            fprintf('Sample %d/%d CHECKED\n', idxGlobalSample, numGlobalSamples);            
            idxGlobalSample = idxGlobalSample + 1;

            % Commented: - Test correctness
%             figure(1);
%             subplot(1,2,1);
%             imshow(patch);
%             axis off; axis image;
%             subplot(1,2,2);
%             imshow(img);
%             rectangle('position', [col row width height], 'LineWidth', 1, 'EdgeColor', [0 0 1]);
%             keyboard;

        end

    end
        
end


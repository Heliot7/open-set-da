function samples = getImgNegativeSamples(input, data, sampleSize, numNegSamples)

    % get always same random samples to reproduce results
    rng(7);

    fprintf('Read negative training samples\n');
    samples = zeros(sampleSize(1),sampleSize(2),3,numNegSamples,'uint8');
    samplesPerImage = 2.5*ceil(numNegSamples / length(data.imgPaths)); % x10 to be faster (less re-read)
    if(~input.isScaleInvariance)
        if(data.classSetup(1).muSize(1) > data.classSetup(1).muSize(2))
            negAR = input.featureInfo.patchSize(1) / data.classSetup(1).muSize(1);
            negStd = data.classSetup(1).stdSize(1) / data.classSetup(1).muSize(1);
        else
            negAR = input.featureInfo.patchSize(2) / data.classSetup(1).muSize(2);
            negStd = data.classSetup(1).stdSize(2) / data.classSetup(1).muSize(2);
        end
        negScales = sort([negAR, max(0.1,negAR-negStd):0.33:negAR+negStd]);
        % negScales = negAR;
        modAR = [];
        if(~input.keepAR)
            for i = 1:length(data.classSetup)
                modAR = [modAR; data.classSetup(i).ar];
                if(isfield(data.classSetup(i),'metadata'))
                    for j = 1:length(data.classSetup(i).metadata)
                        modAR = [modAR; data.classSetup(i).metadata(j).ar];
                    end
                end
            end
        end
    else
        negScales = 1.0;
    end

    % Take valid randomised negative imgs in different scales
    idx = 0;
    while(idx < numNegSamples)

        idxImg = randi(length(data.imgPaths));
        rawImg = imread(data.imgPaths{idxImg});
        resizedImg = cell(length(negScales),1);
        for lvl = 1:length(negScales)
            resizedImg{lvl} = imresize(rawImg,negScales(lvl));
        end
        oldNumScale = 0;
        
        for idxSample = 1:samplesPerImage
            
            % Pick a random scale
            numScale = randi(length(negScales));
            if(numScale ~= oldNumScale)
                img = resizedImg{numScale};
                oldNumScale = numScale;
            end
            if(size(img,1) >= sampleSize(1) + 3 && size(img,2) >= sampleSize(2) + 3)
            
                p0 = floor(rand(1,2).*([size(img,1) size(img,2)] - sampleSize - 3)) + 1;
                BB = [p0(1), p0(2), sampleSize(1)+2, sampleSize(2)+2];
                
                % Check if it is not overlapping 33% or more
                isOverlap = false;
                listObjs = find(data.annotations.imgId == idxImg);
                % imshow(img);
                for idxBB = 1:length(listObjs)
                    BB2 = data.annotations.BB(listObjs(idxBB),:);
                    % Adapt annotation to resizing
                    BB2(1:2) = [size(img,1) size(img,2)] / 2.0 + (BB2(1:2) - [size(rawImg,1) size(rawImg,2)]/2.0) * negScales(numScale);
                    BB2(3:4) = BB2(3:4) * negScales(numScale);
                    isOverlap = isOverlap || detectOverlap(BB, BB2, input.lvlOverlapNegSamples);  
                    %rectangle('position', [BB2(2) BB2(1) BB2(4) BB2(3)]);
                end
                % rectangle('position', [BB(2) BB(1) BB(4) BB(3)]);
                % keyboard;

                % Check if it is not overlapping an ignore region
                if(isfield(data, 'ignore'))
                    listIgnores = find(data.ignore.imgId == idxImg);
                    for idxAnno = 1:length(listIgnores)
                        BB2 = data.ignore.BB(listIgnores(idxAnno),:);
                        % Adapt ignore region to resizing
                        BB2(1:2) = [size(img,1) size(img,2)] / 2.0 + (BB2(1:2) - [size(rawImg,1) size(rawImg,2)]/2.0) * negScales(numScale);
                        BB2(3:4) = BB2(3:4) * negScales(numScale);                        
                        isOverlap = isOverlap || detectOverlapIgnore(BB2, BB);
                    end
                end

                if(~isOverlap)
                    patch = grey2rgb(img(BB(1):BB(1)+BB(3)-1,BB(2):BB(2)+BB(4)-1,:));
                    idx = idx + 1;                    
                    patch = patch(2:end-1,2:end-1,:);
                    samples(:,:,:,idx) = grey2rgb(patch);
                    if(mod(idx,1000) == 0)
                        fprintf('%d/%d negative samples CHECKED\n', idx, numNegSamples);
                    end
                    if(idx >= numNegSamples)
                        break;
                    end
                end

            end

        end

    end
        
end


function input = setupFeatures(input, data)

    % Default sizes per class
    % > Hardcoded for good quality/performance results
    if(strcmp(input.typeDescriptor, 'HOG'))
        % Border: 16+8pxls (8*1.5 for each side)
        % - Daimler 0 border. Otherwise: 16+8pxls
        % -> human = [144 80];
        % -> car = [88 120];
        patchSize = [128, 128];
        featureSize = [round(patchSize./8) - 3 31];
        featureDim = prod(round(patchSize./8) - 3) * 31;
    elseif(~isempty(strfind(input.typeDescriptor,'CNN')))
        if(~isempty(strfind(input.typeDescriptor,'fc7')))
            if(~isempty(strfind(input.cnnName,'AlexNet')))
                patchSize = [227 227];
            elseif(~isempty(strfind(input.cnnName,'VGG')))
                patchSize = [224 224]; % [224 224];
            end
        elseif(~isempty(strfind(input.typeDescriptor,'pool5')))
            if(~isempty(strfind(input.cnnName,'AlexNet')))
                patchSize = [227 227];
            elseif(~isempty(strfind(input.cnnName,'VGG')))
                patchSize = [224 224]; % [224 224];
            end
        elseif(~isempty(strfind(input.typeDescriptor,'conv5')))
            patchSize = [227 227];
        end
        if(~isprop(input.sourceDataset,'source'))
            [featureDim, featureSize] = getConvDims(input.typeDescriptor, patchSize);
        else
            featureDim = 4096;
            featureSize = [1, 1, 4096];
        end
    elseif(strcmp(input.typeDescriptor, 'BoW'))
        patchSize = [1 1];
        featureSize = 800; % 800
        featureDim = featureSize;
    else
        patchSize = [1 1];
        featureSize = 1; % 800
        featureDim = featureSize;
    end
    addprop(input,'featureInfo');
    input.featureInfo = struct('patchSize', patchSize, 'featureSize', featureSize, 'featureDim', featureDim, 'classes', []);

    % Classes specifc features
    input.featureInfo.classes = repmat(struct('patchSize',''), length(data.classSetup), 1);
    for c = 1:length(input.featureInfo.classes)
        if(strcmp(input.typeDescriptor, 'HOG'))
            if(strcmpi(data.classSetup(c).className,'human'))
                patchSize = [144 80];
            elseif(strcmpi(data.classSetup(c).className,'car'))
                patchSize = [88 120];
            else
                patchSize = [128 128];
            end
            if(input.autoBB)
                patchSize = setupSize(data.classSetup(c).muSize);
            end
            featureSize = [round(patchSize./8) - 3 31];
            featureDim = prod(round(patchSize./8) - 3) * 31;
        elseif(~isempty(strfind(input.typeDescriptor,'CNN')))
            if(input.autoBB)
                patchSize = setupSize(data.classSetup(c).muSize);
            else
                if(~isempty(strfind(input.typeDescriptor,'fc7')))
                    if(~isempty(strfind(input.cnnName,'AlexNet')))
                        patchSize = [227 227];
                    elseif(~isempty(strfind(input.cnnName,'VGG')))
                        patchSize = [224 224]; % [224 224];
                    end
                elseif(~isempty(strfind(input.typeDescriptor,'pool5')))
                    patchSize = [227 227];
                elseif(~isempty(strfind(input.typeDescriptor,'conv5')))
                    patchSize = [227 227];
                end
            end
            if(~isprop(input.sourceDataset,'source'))
                [featureDim, featureSize] = getConvDims(input.typeDescriptor, patchSize);
            else
                featureDim = 4096;
                featureSize = [1, 1, 4096];
            end
        end
        input.featureInfo.classes(c).patchSize = patchSize;
        input.featureInfo.classes(c).featureSize = featureSize;
        input.featureInfo.classes(c).featureDim = featureDim;
        % Add feature size and dim if ever used own dims
        if(isfield(data.classSetup(c),'metadata'))
            input.featureInfo.classes(c).metadata = repmat(struct('patchSize',''), length(data.classSetup(c).metadata), 1);
            for i = 1:length(data.classSetup(c).metadata)
                if(~input.autoBB)
                    input.featureInfo.classes(c).metadata(i).patchSize = input.featureInfo.classes(c).patchSize;
                    input.featureInfo.classes(c).metadata(i).featureSize = input.featureInfo.classes(c).featureSize;
                    input.featureInfo.classes(c).metadata(i).featureDim = input.featureInfo.classes(c).featureDim;
                else
                    patchSize = setupSize(data.classSetup(c).metadata(i).muSize);
                    if(strcmp(input.typeDescriptor, 'HOG'))
                        featureSize = [round(patchSize./8) - 3 31];
                        featureDim = prod(round(patchSize./8) - 3) * 31;
                    elseif(~isempty(strfind(input.typeDescriptor,'CNN')))
                         if(~isprop(input.sourceDataset,'source'))
                            [featureDim, featureSize] = getConvDims(input.typeDescriptor, patchSize);
                        else
                            featureDim = 4096;
                            featureSize = [1, 1, 4096];
                        end
                    end
                    input.featureInfo.classes(c).metadata(i).patchSize = patchSize;
                    input.featureInfo.classes(c).metadata(i).featureSize = featureSize;
                    input.featureInfo.classes(c).metadata(i).featureDim = featureDim;
                end  
            end
        end
    end
    
end

function patchSize = setupSize(mu)

    muProd = prod(mu);
    maxProd = 75000;
    if(muProd < maxProd)
        patchSize = round(mu);
    else
        patchSize = round(mu * maxProd / muProd);
    end

end

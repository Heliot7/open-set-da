function [idLabelsStr, nameLabels, metadata] = getIdLabels(dataset_info, dataAnnotations, in_metadata)

    cLabels = ones(length(dataAnnotations.classes),1);
    vpLabels = ones(length(dataAnnotations.classes),1);
    metadata = [];
    if(isprop(dataset_info,'classes') && isfield(dataAnnotations,'classes'))
        cLabels = getClassId(dataset_info.classes, dataAnnotations.classes);
        metadata = [metadata; {'classes'}];
    end
    if(isprop(dataset_info,'azimuth') && isfield(dataAnnotations,'vp'))
        vpLabels = getAzimuthId(dataset_info.azimuth, dataAnnotations.vp.azimuth);
        metadata = [metadata; {'azimuth'}];
    end
    if(nargin > 2)
        isOk = ismember(metadata,in_metadata);
        metadata(~isOk) = [];
    end

    % Estimate ids for a joint combination
    % TODO: Add Generic meta-data (subclasses, unique attributes, etc.)
    jumpId = 1;
    idLabels = ones(length(dataAnnotations.classes),1);
    numAttributes = length(metadata);
    for idxMD = numAttributes:-1:1

        strLabel = metadata{idxMD};
        if(strcmpi(strLabel,'classes'))
            values = cLabels;
            listLabels = dataset_info.classes;
        elseif(strcmpi(strLabel,'azimuth'))
            values = vpLabels;
            listLabels = dataset_info.azimuth;
        end
        idLabels = idLabels + jumpId*(values-1);
        jumpId = jumpId * length(listLabels);
    end

    % Meta-data information: correspondences to Id's
    numIdLabels = jumpId;
    nameLabels = [];
    if(~isempty(metadata))
        nameLabels = rec_hashLabels(dataset_info, metadata, 1, cell(numIdLabels, numAttributes), 1, ones(numAttributes,1));
    end
    
    % Translate ids
    idLabelsStr = cell(length(idLabels),size(nameLabels,2));
    for idxLabel = 1:size(nameLabels,1)
        isLabel = (idLabels == idxLabel);
        idLabelsStr(isLabel,:) = repmat(nameLabels(idxLabel,:),[sum(isLabel) 1]);
    end
    
end

function [hashLabels, idxHash] = rec_hashLabels(dataset_info, metaData, idxLabel, hashLabels, idxHash, idxMetaData)

    for idxData = 1:length(dataset_info.(metaData{idxLabel}))
        
        idxMetaData(idxLabel) = idxData;
        
        % Base case
        if(idxLabel == length(metaData))
            
            for i = 1:length(metaData)
                labels = dataset_info.(metaData{i});
                if(iscell(labels))
                    value = labels{idxMetaData(i)};
                elseif(ismatrix(labels))
                     value = num2str(labels(idxMetaData(i)));
                end
                hashLabels{idxHash, i} = value;
            end
            idxHash = idxHash + 1;
            
        else % Recursivity
            
            [hashLabels, idxHash] = rec_hashLabels(dataset_info, metaData, idxLabel+1, hashLabels, idxHash, idxMetaData);
            
        end
        
    end

end



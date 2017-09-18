function [sFeat, tFeat, W, classifiers, transferIds, testFeat] = DA(input, metadata, sData, sClasses, sLabels, sFeat, tData, tClasses, tLabels, tFeat, testData, testLabels, testFeat)

    % Assign class numbers to each 
    sIds = zeros(size(sLabels,1), 1);
    for i = 1:size(sClasses,1)
        sIds(size(sLabels,2) == sum(ismember(sLabels,sClasses(i,:)),2)) = i;
    end
    tIds = zeros(size(tLabels,1), 1);
    testIds = zeros(size(testLabels,1), 1);
    for j = 1:size(tClasses,1)
       posSameId = find(size(sClasses,2) == sum(ismember(sClasses,tClasses(j,:)),2), 1);
       if(~isempty(posSameId))
           tIds(size(tLabels,2) == sum(ismember(tLabels,tClasses(j,:)),2)) = posSameId;
           if(~isempty(testLabels))
            testIds(size(testLabels,2) == sum(ismember(testLabels,tClasses(j,:)),2)) = posSameId;
           end
       else % new Id
           i = i+1;
           tIds(size(tLabels,2) == sum(ismember(tLabels,tClasses(j,:)),2)) = i;
           if(~isempty(testLabels))
            testIds(size(testLabels,2) == sum(ismember(testLabels,tClasses(j,:)),2)) = i;
           end
       end
    end
    
    % Case 4 viewpoint refinement
    if(input.is4ViewSupervised)
        ids4ViewsSrc = ones(length(sData.annotations.imgId),1);
        ids4ViewsTgt = ones(length(tData.annotations.imgId),1);
        if(input.is4ViewSupervised && ismember(input.typeDA,{'gfk','CORAL'}))
            ids4ViewsSrc = get4ViewIds(input, sData);
            ids4ViewsTgt = get4ViewIds(input, tData);
        end
    end
    %
    
    W = []; classifiers = []; transferIds = [];
    switch input.typeDA
        case {'translation', 'scale', 'transScale'}
            sFeat = DA_transScale(input.typeDA, input, sData, sIds, sFeat, tData, tFeat);
        case {'ATI', 'corr'}
            [sFeat, tFeat, W, transferIds, testFeat, classifiers] = ...
                DA_ATI(input, sData, sIds, sFeat, tData, tIds, tFeat, input.typeDA, testData, testIds, testFeat);
        case 'whitening'
            sFeat = whiten(sFeat')';
            % - Best results by only whitening source data
%             tFeat = whiten(tFeat')';
        case 'gfk'
            if(input.is4ViewSupervised)
                numIter = max(ids4ViewsSrc);
                for i = 1:numIter
                    [sFeat(ids4ViewsSrc == i,:), tFeat(ids4ViewsTgt == i,:), W] = ...
                        DA_GFK(sFeat(ids4ViewsSrc == i,:), tFeat(ids4ViewsTgt == i,:), input.dimPCA);
                end
            else
                [sFeat, tFeat, W] = DA_GFK(sFeat, tFeat, input.dimPCA);
            end
        case 'TCA'
            maskLabels = logical([ones(size(sFeat,1),1);zeros(size(tFeat,1),1)]);
            [allFeat, W] = DA_TCA([sFeat; tFeat], maskLabels, sIds, maskLabels, input.dimPCA);
            sFeat = allFeat(1:size(sFeat,1),:);
            tFeat = allFeat(size(sFeat,1)+1:end,:);
        case 'SA'
            if(input.is4ViewSupervised)
                sFeat_aux = zeros(size(sFeat,1),input.dimPCA);
                tFeat_aux = zeros(size(tFeat,1),input.dimPCA);
                numIter = max(ids4ViewsSrc);
                for i = 1:numIter
                    [sFeat_aux(ids4ViewsSrc == i,:), tFeat_aux(ids4ViewsTgt == i,:), W] = ...
                        DA_SA(sFeat(ids4ViewsSrc == i,:), tFeat(ids4ViewsTgt == i,:), input.dimPCA);
                end
                sFeat = sFeat_aux;
                tFeat = tFeat_aux;
            else
                transferIds = DA_SA_libsvm(input, sIds, sFeat, tIds, tFeat, input.dimPCA);
            end
        case 'MMDT' % Revise
            [~, ~, tFeat, W] = MMDT(input, sData, sFeat, sIds, tData, tFeat, tIds);
        case 'CORAL'
            if(input.is4ViewSupervised)
                numIter = max(ids4ViewsSrc);
                % transferIds = zeros(length(ids4ViewsTgt),1);
                for i = 1:numIter
                    [sFeat(ids4ViewsSrc == i,:), tFeat(ids4ViewsTgt == i,:), W] = ...
                        DA_Coral(sFeat(ids4ViewsSrc == i,:), tFeat(ids4ViewsTgt == i,:));
    %                 transferIds(ids4ViewsTgt == i) = DA_Coral_libsvm(input, sIds(ids4ViewsSrc == i,:), ...
    %                     sFeat(ids4ViewsSrc == i,:), tIds(ids4ViewsTgt == i,:), tFeat(ids4ViewsTgt == i,:));
                end
            else
                transferIds = DA_Coral_libsvm(input, sIds, sFeat, tIds, tFeat);
            end
        otherwise
            error('[Caught ERROR: Wrong Domain Adaptation method: %s]', input.typeDA);
    end

    if(~isempty(transferIds))
        newLabels = cell(size(tFeat,1),size(sClasses,2));
        for idxLabel = 1:size(sClasses,1)
            isLabel = (transferIds == idxLabel);
            if(~isempty(isLabel))
                newLabels(isLabel,:) = repmat(sClasses(idxLabel,:),[sum(isLabel) 1]);
            end
        end
        transferIds = newLabels;
    end
    
end

function ids4Views = get4ViewIds(input, data)

    lowerBound = [315, 45, 135, 225];
    upperBound = [45, 135, 225, 315];                
    angles = data.annotations.vp.azimuth;
    ids4Views = zeros(length(angles),1);
    stepSize = (360/length(input.sourceDataset.azimuth))/2;
    listViewpoints = input.sourceDataset.azimuth;
    if(length(listViewpoints) == 36)
        stepSize = 0;
    end       
    for i = 1:4 % front, rear, left, right views
        if(i == 1) % front view -> special case
            samples = angles > lowerBound(i) + stepSize | angles < upperBound(i) - stepSize;
        elseif(i == 3)
            samples = angles > lowerBound(i) + stepSize & angles < upperBound(i) - stepSize;
        else
            samples = angles >= lowerBound(i) - stepSize & angles <= upperBound(i) + stepSize;
        end
        ids4Views(samples) = i;
    end

end


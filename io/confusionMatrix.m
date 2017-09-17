function [results, angleErr, stdDevAngleErr] = confusionMatrix(input, metaData, strClasses, gtData, tgtClasses, gtLabels, transferLabels, mDir, mFile)

    if(nargin < 9)
        mFile = '';
    end
    listClasses = unique([input.sourceDataset.classes, input.targetDataset.classes]);
    results = [];
    angleErr = +Inf*ones(length(listClasses),2);
    stdDevAngleErr = +Inf*ones(length(listClasses),2);
        
    % Compute probabilities for each class
    resC = transferLabels(:,strcmpi(metaData,'classes'));
    gtC = gtLabels(:,strcmpi(metaData,'classes'));
    if(length(listClasses) > 1)
        probMat = zeros(length(listClasses));
        elemMat = zeros(length(listClasses));
        for idxRow = 1:length(listClasses)
            isGtClass = ismember(gtC,listClasses{idxRow});
            numElems = sum(isGtClass);                
            for idxCol = 1:length(listClasses)
                isResClass = ismember(resC,listClasses{idxCol});
                matches = sum(prod([isGtClass, isResClass],2));
                probMat(idxRow, idxCol) = matches / numElems;
                elemMat(idxRow, idxCol) = matches;
            end
        end
        % For drawing empty classes in a different way
        probMat(isnan(probMat)) = -1;
        file = [mDir ' ' mFile];
        drawConfusionMatrix(length(listClasses), elemMat, probMat, input.targetDataset.classes, file, 'class', '', 'pr');
        drawConfusionMatrix(length(listClasses), elemMat, probMat, input.targetDataset.classes, file, 'class', '', 'el');
        % Store results
        accuracy = sum(diag(elemMat))/sum(sum(elemMat))*100.0;
        results = ones(1,2);
        results(1,1) = round(accuracy*100.0)/100.0;
        emptyElems = sum(diag(probMat) == -1);
        results(1,2) = (sum(diag(probMat))+emptyElems)/(length(listClasses)-emptyElems)*100.0;
    end
    
    if(input.isOpenset || input.isWSVM) % To compare 10vs10 classes
        listClasses(end) = [];
        probMat = zeros(length(listClasses));
        elemMat = zeros(length(listClasses));
        for idxRow = 1:length(listClasses)
            isGtClass = ismember(gtC,listClasses{idxRow});
            numElems = sum(isGtClass);                
            for idxCol = 1:length(listClasses)
                isResClass = ismember(resC,listClasses{idxCol});
                matches = sum(prod([isGtClass, isResClass],2));
                probMat(idxRow, idxCol) = matches / numElems;
                elemMat(idxRow, idxCol) = matches;
            end
        end
        % For drawing empty classes in a different way
        probMat(isnan(probMat)) = -1;
        file = [mDir ' ' mFile];
        drawConfusionMatrix(length(listClasses), elemMat, probMat, input.targetDataset.classes, file, 'class', '', 'pr10');
        drawConfusionMatrix(length(listClasses), elemMat, probMat, input.targetDataset.classes, file, 'class', '', 'el10');
        % Store results
        accuracy = sum(diag(elemMat))/sum(sum(elemMat))*100.0;
        results = ones(1,2);
        results(1,1) = round(accuracy*100.0)/100.0;
        emptyElems = sum(diag(probMat) == -1);
        results(1,2) = (sum(diag(probMat))+emptyElems)/(length(listClasses)-emptyElems)*100.0;        
    end

    % - Label-class independent confusion matrix (e.g. viewpoints)
    for i = 2:length(metaData)
    
        strLabel = metaData{i};
        res = transferLabels(:,i);
        gt = gtLabels(:,i);
        for idxClass = 1:length(listClasses)
            
            strClass = listClasses{idxClass};
            isResClass = ismember(resC,strClass);
            isGtClass = ismember(gtC,strClass);
            labels = input.targetDataset.(strLabel);
            if(~iscell(labels))
                labels = cellfun(@num2str, num2cell(labels), 'UniformOutput', false);
            end
            numLabels = length(labels);
            probMat = zeros(numLabels);
            elemMat = zeros(numLabels);
            for idxRow = 1:numLabels
                idxGtElems = isGtClass & ismember(gt, labels{idxRow});
                numElems = sum(idxGtElems);
                for idxCol = 1:numLabels
                    idxDetElems = isResClass & ismember(res, labels{idxCol});
                    matches = sum(prod([idxGtElems, idxDetElems],2));
                    probMat(idxRow, idxCol) = matches / numElems;
                    elemMat(idxRow, idxCol) = matches;
                end
            end
            % For drawing empty classes in a different way
            probMat(isnan(probMat)) = -1;
            
            file = [mDir ' ' mFile];
            drawConfusionMatrix(numLabels, elemMat, probMat, labels, file, strClass, strLabel, 'pr');
            drawConfusionMatrix(numLabels, elemMat, probMat, labels, file, strClass, strLabel, 'el');
            if(strcmpi(strLabel,'azimuth'))
                numGT = cell2mat(cellfun(@str2num, gt, 'UniformOutput', false));
                drawAzimuthResults(input.targetDataset.(strLabel), strClass, numGT(isGtClass), diag(probMat), file);
%                 [angleErr(idxClass,1), stdDevAngleErr(idxClass,1)] = ...
%                     angleError(input.targetDataset.(strLabel), elemMat, file, input.is4ViewSupervised, 'elem');
%                 [angleErr(idxClass,2), stdDevAngleErr(idxClass,2)] = ...
%                     angleError(input.targetDataset.(strLabel), elemMat, file, input.is4ViewSupervised, 'prob');
                gtViewpoints = gtData.annotations.vp.azimuth;
                numTransferred = cell2mat(cellfun(@str2num, res, 'UniformOutput', false));
                [angleErr, stdDevAngleErr] = angleErrorGT(gtViewpoints(isGtClass), numTransferred(isGtClass), file, strClass, input.is4ViewSupervised);
            end
            accuracyEl = sum(diag(elemMat))/sum(sum(elemMat))*100.0;
            emptyElems = sum(diag(probMat) == -1);
            accuracyPr = (sum(diag(probMat))+emptyElems)/(numLabels-emptyElems)*100.0;
            results = [results; [round(accuracyEl*100.0)/100.0 accuracyPr]];
        end
        
    end
    
    % Combined confusion matrix of all attributes
    % > Remove class if known
    allClasses = [strClasses; tgtClasses];
    if(size(allClasses,2) > 1)
        [~, I1] = unique(allClasses(:, 2));
        I1 = sort(I1);
        allClasses = allClasses(I1, :);
        if(input.isClassSupervised || length(listClasses) < 2)
            allClasses(:,strcmpi(metaData,'classes')') = [];
            gt = gtLabels; res = transferLabels;
            gt(:,strcmpi(metaData,'classes')') = [];
            res(:,strcmpi(metaData,'classes')') = [];
        end
        if(size(allClasses,1) > 1 && size(allClasses,2) > 1)
            numLabels = size(allClasses,1);
            probMat = zeros(numLabels);
            elemMat = zeros(numLabels);
            for idxRow = 1:numLabels
                idxGtElems = ismember(gt, allClasses{idxRow});
                numElems = sum(idxGtElems);
                for idxCol = 1:numLabels
                    idxDetElems = ismember(res, allClasses{idxCol});
                    matches = sum(prod([idxGtElems, idxDetElems],2));
                    probMat(idxRow, idxCol) = matches / numElems;
                    elemMat(idxRow, idxCol) = matches;
                end
            end
            probMat(isnan(probMat)) = -1;
            file = [mDir ' ' mFile];
            strClasses = allClasses(:,1);
            for i = 2:size(allClasses,2)
                strClasses = strcat(strClasses, {' '}, allClasses(:,i));
            end
            drawConfusionMatrix(numLabels, elemMat, probMat, strClasses, file, 'All', '', 'pr');
            drawConfusionMatrix(numLabels, elemMat, probMat, strClasses, file, 'All', '', 'el');
            accuracyEl = sum(diag(elemMat))/sum(sum(elemMat))*100.0;
            emptyElems = sum(diag(probMat) == -1);
            accuracyPr = (sum(diag(probMat))+emptyElems)/(numLabels-emptyElems)*100.0;
            results = [results; [round(accuracyEl*100.0)/100.0 accuracyPr]];
        end
    end
end



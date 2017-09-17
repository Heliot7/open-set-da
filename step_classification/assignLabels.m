function [newLabels, scores] = assignLabels(input, classifiers, srcClasses, srcMetadata, tgtIds, features, isVerbose)

    if(nargin < 7)
        isVerbose = true;
    end  

    numSamples = size(features,1);
    numClassifiers = size(classifiers.w_pos,1);    
    % Precompute normalisation scores
    normal = zeros(1, numClassifiers);
    for i = 1:numClassifiers
        template = classifiers.w_pos(i,:) - classifiers.w_neg(i,:);
        normal(i) = sum(template.*template);
    end
    
    if(isVerbose)
        fprintf('Computing scores labels\n');
    end
    scores = (features * (classifiers.w_pos - classifiers.w_neg)' ...
        + repmat(classifiers.bias', [numSamples 1])) ./ repmat(normal, [numSamples 1]);

    if(strcmpi(input.methodSVM,'liblinear'))
        classifier2object = 1:numClassifiers;
    elseif(strcmpi(input.methodSVM,'libsvm'))
        classifier2object = zeros(numClassifiers,1);
        classifier2negObj = zeros(numClassifiers,1);
        numClasses = round((1 + sqrt(1 + 4*2*numClassifiers)) / 2);
        idxClassifier = numClasses - 1;
        maxClassifier = 1;
        for i = 1:numClasses-1
            classifier2object(maxClassifier:maxClassifier+idxClassifier-1) = i;
            classifier2negObj(maxClassifier:maxClassifier+idxClassifier-1) = i+1:numClasses;
            maxClassifier = maxClassifier + idxClassifier;
            idxClassifier = idxClassifier - 1;
        end
        
    end
    
    % Class labels are given in the target samples
    listClasses = input.sourceDataset.classes;
    if(input.isClassSupervised && sum(strcmpi(srcMetadata,'classes')) || length(srcMetadata) > 1)
        for idxClass = 1:length(listClasses)
            
            isTgtClass = sum(ismember(tgtIds,listClasses{idxClass}),2,'native');
            isSrcClass = sum(ismember(srcClasses,listClasses{idxClass}),2,'native');
            if(strcmpi(input.methodSVM,'liblinear'))
                notSrc = ismember(classifier2object,find(~isSrcClass));
            elseif(strcmpi(input.methodSVM,'libsvm'))
                notSrc = ismember(classifier2object,find(~isSrcClass)) & ismember(classifier2negObj,find(~isSrcClass));
            end
            if(isempty(strfind(input.methodSVM,'libsvm')))
                scores(isTgtClass,notSrc) = -Inf;
            end
        end
    end
    % If viewpoint refinement: ignore scores out of coarse viewpoint
    if(input.is4ViewSupervised && sum(strcmpi(srcMetadata,'azimuth')))

        if(strcmpi(input.methodSVM,'libsvm'))
            [~, ~, ~, scores] = evalc('svmpredict(zeros(size(features,1),1), double(features), classifiers.model)');
        end
        
        lowerBound = [315, 45, 135, 225];
        upperBound = [45, 135, 225, 315];
        listViewpoints = input.sourceDataset.azimuth;
        strViewpoints = srcClasses(:,strcmpi(srcMetadata,'azimuth'));
        for idxClass = 1:length(listClasses)
            for idxVP = 1:length(listViewpoints)

                angle = listViewpoints(idxVP);
                % angle = listViewpoints(idLabels(idxImg));
                view = false(4,1);
                for v = 1:4
                    if((v == 1) && (angle > lowerBound(v) || angle < upperBound(v)))
                        view(v) = true;
                    elseif((v == 3) && (angle > lowerBound(v) && angle < upperBound(v)))
                        view(v) = true;
                    elseif((v == 2 || v == 4) && (angle >= lowerBound(v) && angle <= upperBound(v)))
                        view(v) = true;
                    end
                end
                % Selection range of feasible candidates from syn classifiers
                % Front (1) and back (3) views less dominant if views inbetween
                if(view(1)) % front view -> special case
                    classCandidates = find(listViewpoints > lowerBound(1) | listViewpoints < upperBound(1));
                elseif(view(3))
                    classCandidates = find(listViewpoints > lowerBound(view) & listViewpoints < upperBound(view));
                else % left (2) and right (4) views
                    classCandidates = find(listViewpoints >= lowerBound(view) & listViewpoints <= upperBound(view));
                end
                selectedViews = strViewpoints(ismember(1:length(listViewpoints),classCandidates));
                selectedIds = retrieveId(srcMetadata, srcClasses, 'classes', listClasses(idxClass), 'azimuth', selectedViews);
                selectedTargetSamples = prod(ismember(tgtIds,selectedIds),2,'native');
                if(strcmpi(input.methodSVM,'liblinear'))
                    selectedSourceClassifiers = ~prod(ismember(srcClasses,selectedIds),2,'native');
                elseif(strcmpi(input.methodSVM,'libsvm'))
                    discardLbls = find(~prod(ismember(srcClasses,selectedIds),2,'native'));
                    selectedLbls = find(prod(ismember(srcClasses,selectedIds),2,'native'));
                    selectedSourceClassifiers = ismember(classifier2object,discardLbls) | ismember(classifier2negObj,discardLbls);
                    if(length(selectedLbls) == 1) % if only 1 candidate, choose it!
                        selectedSourceClassifiers(classifier2object == selectedLbls) = false; 
                    end
                end
                scores(selectedTargetSamples,selectedSourceClassifiers) = -Inf;
            end
        end
    end
    
    % Convert back from identifiers to proper meaningful labels with scores
    newLabels = cell(numSamples,size(srcClasses,2));
    if(~input.is4ViewSupervised && ~isempty(strfind(input.methodSVM,'libsvm')))
        if(strcmpi(input.methodSVM,'libsvm')) % CASE from LIBSVM
            % [prob, ids, lala] = svmpredict(zeros(size(features,1),1), double(features), classifiers.model);
            [~, ids] = evalc('svmpredict(zeros(size(features,1),1), double(features), classifiers.model)');
        elseif(strcmpi(input.methodSVM,'libsvm-open')) % CASE from LIBSVM-OPEN
            ids = svmpredict_open(2*ones(size(features,1),1), double(features), classifiers.model, classifiers.model_open); % , '-P 0.1 -C 0.001');
            if(input.isWSVM)
                ids(ids == -99999) = 11;
            end
            % [~, ids] = evalc('svmpredict_open(zeros(size(features,1),1), double(features), classifiers.model)');
        end
        for idxLabel = 1:size(srcClasses,1)
            isLabel = (ids == idxLabel);
            if(~isempty(isLabel))
                newLabels(isLabel,:) = repmat(srcClasses(idxLabel,:),[sum(isLabel) 1]);
            end
        end
    elseif(strcmpi(input.methodSVM,'liblinear'))
        [~, idxs] = max(scores,[],2);
        for idxLabel = 1:size(srcClasses,1)
            isLabel = (idxs == idxLabel);
            if(~isempty(isLabel))
                newLabels(isLabel,:) = repmat(srcClasses(idxLabel,:),[sum(isLabel) 1]);
            end
        end
    % OVO: Voting scheme and selected the one with highest +1's
    elseif(strcmpi(input.methodSVM,'libsvm'))
        for idxSample = 1:numSamples
            voting = zeros(numClasses,1);
            winnersPos = classifier2object(scores(idxSample,:) > 0);
            winnersNeg = classifier2negObj(scores(idxSample,:) <= 0 & scores(idxSample,:) ~= -Inf);
            for idxClass = 1:numClasses
               voting(idxClass) = sum(winnersPos == idxClass) + sum(winnersNeg == idxClass);
            end
            [~, idx] = max(voting);
            % Take first one and done... (could be improved :P)
            if(length(idx) > 1)
                idx = idx(1);
            end
            newLabels(idxSample,:) = srcClasses(idx,:);
        end
    end
    
end


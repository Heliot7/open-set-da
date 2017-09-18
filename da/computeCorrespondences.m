function [sCorr, tCorr, selDist, sCorrAll, tCorrAll, allDist, th] = computeCorrespondences(input, itOut, it, NN, ...
    sData, sFeat, sCentroids, sCenIds, sAR, sBlocks, tData, tFeat, tCentroids, tCenIds, tAR, tBlocks, testFeat)

    % Remove background data from src if exists
    if(~input.includeBgClass)
        isBgClass = ismember(input.sourceDataset.classes,'zz_unknown');
        sCentroids(isBgClass,:) = [];
        if(input.isClassSupervised && strcmpi(input.typeWildSupervision,'100'))
            bgCentroids = find(find(isBgClass) == tBlocks);
            tCentroids(find(isBgClass) == tBlocks,:) = [];
            if(~isempty(NN))
                NN(find(isBgClass) == tBlocks) = [];
                NN(NN > max(bgCentroids)) = NN(NN > max(bgCentroids)) - length(bgCentroids);
            end
            tBlocks(find(isBgClass) == tBlocks) = [];
        end
    end
    
    % Supervised approach, where unlabelled test data takes part
    fixedIds = [];
    if(~input.is4ViewSupervised && input.isClassSupervised && ~(itOut == 1&& it == 1))
        tCentroids = [tCentroids; testFeat];
        fixedIds = tBlocks;
        % In the wild case supervised
        if(~input.includeBgClass)
%             isBgClassTgt = ismember(tData.annotations.classes,'background');
%             tCentroids(isBgClassTgt,:) = [];
            fixedIds(fixedIds > find(isBgClass)) = fixedIds(fixedIds > find(isBgClass)) - 1;
        end
    end

    numSrcClusters = size(sCentroids,1);    
    numTgtClusters = size(tCentroids,1);
    fprintf('Reassign connections between src(%d)/tgt(%d) clusters\n', numSrcClusters, numTgtClusters);
    
    % Add BBs AR in the distances
    % compute median distance
    % distances = medianDist(sCentroids, sBlocks, tCentroids, tCenIds);        
    distancesL2 = zeros(numTgtClusters, numSrcClusters, 'single');
%     tARnorm = (tAR - min(tAR)) ./ (max(tAR) - min(tAR));
%     sARnorm = (sAR - min(sAR)) ./ (max(sAR) - min(sAR));
    for c = 1:numTgtClusters
        dist = repmat(tCentroids(c,:),[numSrcClusters,1]) - sCentroids;
        dist = sqrt(sum(dist.*dist, 2)); % * 0.9 + abs(tARnorm(c) - sARnorm)*mean(mean(dist))*0.1;
        distancesL2(c,:) = dist;
    end
    distances = distancesL2;
    distSrc = zeros(numSrcClusters, numSrcClusters, 'single');
    for c = 1:numSrcClusters
        dist = repmat(sCentroids(c,:),[numSrcClusters,1]) - sCentroids;
        dist = sqrt(sum(dist.*dist, 2));
        distSrc(c,:) = dist;
    end
    
    % 4 Connections with distance 0 (semi-supervised)
    if(input.is4ViewSupervised)

        sCorr = []; tCorr = [];
        angles = input.sourceDataset.azimuth;
        lowerBound = [315, 45, 135, 225];
        upperBound = [45, 135, 225, 315];
        accumTgt = 0;
        for c = 1:length(input.sourceDataset.classes)
            
            idClassSrc = ismember(sData.annotations.classes, input.sourceDataset.classes{c});
            idClassTgt = ismember(tData.annotations.classes, input.sourceDataset.classes{c});
        
            % Number of subclasses per supervised view
            for i = 1:4

                if(i == 1) % front view -> special case
                    srcLabelMatches = find(angles > lowerBound(i) | angles < upperBound(i));
                elseif(i == 3)
                    srcLabelMatches = find(angles > lowerBound(i) & angles < upperBound(i));
                else
                    srcLabelMatches = find(angles >= lowerBound(i) & angles <= upperBound(i));
                end
                % Adapt to class position (if > 1)
                srcLabelMatches = srcLabelMatches + (c-1)*length(angles);
                idViews = (tBlocks == i+4*(c-1));
                classDistances = distances(idViews, srcLabelMatches);
                if(~isempty(classDistances))            

                    % Hungarian algorithm
                    if(size(classDistances,1) >= size(classDistances,2))
                        isSrc = true;
                    else
                        isSrc = false;
                        classDistances = classDistances';
                    end

                    typeAssignment = 'Hungarian'; % 'MILP' 'Hungarian'
                    % Compute optimal assignaments
                    if(strcmpi(typeAssignment, 'MILP'))
                        classSrcDist = distSrc(srcLabelMatches,srcLabelMatches);
                        if(input.numNN >= 1)
                            classNN = NN(idViews,:) - accumTgt;
                        else
                            classNN = [];
                        end
                        [sClassCorr, tClassCorr] = assignmentProblem_NN_MILP(classDistances, classSrcDist, classNN, input.numCorr, input.numLambda, fixedIds);
                        [tClassCorr, idx] = sort(tClassCorr);
                        sClassCorr = sClassCorr(idx);
                    elseif(strcmpi(typeAssignment,'Hungarian'))
                        % Copy extra distances to assign more than one tgt cluster to src's
                        % repetition is possible allowing more flexibility)
                        classDistancesMul = repmat(classDistances, [1, input.numCorr*ceil(size(classDistances,1)./size(classDistances,2))]);
                        sClassCorr = lapjv(classDistancesMul)'; % Update with "assignmentProb"
                        % Give the proper src/tgt cluster id
                        if(isSrc)
                            sClassCorr = mod(sClassCorr-1, length(srcLabelMatches))+1;
                            tClassCorr = (1:length(sClassCorr))';
                        else
                            sClassCorr = mod(sClassCorr-1, sum(idViews))+1;
                            tClassCorr = sClassCorr;
                        end
                        % Optimisation
                        % [sClassCorr, tClassCorr] = assignmentProblem(classDistances);
                    end
                    aux_sClassCorr = sClassCorr;
                    for idxView = 1:length(srcLabelMatches)
                        sClassCorr(aux_sClassCorr == idxView) = srcLabelMatches(idxView);
                    end

                    sCorr = [sCorr; sClassCorr];
                    tCorr = [tCorr; tClassCorr + accumTgt];
                end
                accumTgt = accumTgt + sum(idViews);

            end
        end
        selDist = zeros(length(tCorr),1);
        for i = 1:length(tCorr)
            selDist(i) = distances(tCorr(i), sCorr(i));
        end
        allDist = selDist;
        sCorrAll = sCorr; tCorrAll = tCorr;
        th_dist = 1.0; th = 1.0;
        if(th_dist < 1.0) % be careful for DA-CORR method (must be th=1!)
            [~, idxDist] = sort(selDist);
            toKeep = idxDist(1:round(th_dist*length(selDist)));
            th = length(toKeep) / length(selDist);
            sCorr = sCorr(toKeep);
            tCorr = tCorr(toKeep);
            selDist = selDist(toKeep);
        end
        
    else
        
        % Parameter setup
        maxdistST = max(distances(:)); mindistST = min(distances(:));
        maxdistSS = max(distSrc(:)); mindistSS = min(distSrc(:));
        if(input.numLambda < 1.0)
            if(input.numLambda == 0) % median
                dist_lambdaST = median(distances(:));
            elseif(input.numLambda < 0) % 0 weight
                dist_lambdaST = 0.0;
            else
                dist_lambdaST = input.numLambda*(maxdistST-mindistST) + mindistST;
            end
            % - Add empty nodes => 1 extra Source cluster with distance 0
            distances = [distances, dist_lambdaST*ones(size(distances,1),1)];
            % - For source clusters
            if(input.numLambda == 0) % median
                dist_lambdaSS = median(distSrc(:));
            else
                dist_lambdaSS = 0; % input.numLambda*(maxdistSS-mindistSS) + mindistSS;
            end
            distSrc = [distSrc, dist_lambdaSS*ones(size(distSrc,1),1)];
            distSrc = [distSrc; dist_lambdaSS*ones(1,size(distSrc,2))];
        else
            dist_lambdaST = max(distances(:));
        end

        % Proceed with the assignment problem
        [sCorr, tCorr] = assignmentProblem_NN_MILP(distances, distSrc, NN, input.numCorr, input.numLambda, fixedIds);
        % [sCorr, tCorr] = assignmentProblem_NN_BP(distances, distSrc, NN, input.numCorr, input.numLambda);
        
        % Process correspondences
        selDist = zeros(length(tCorr),1);
        for i = 1:length(tCorr)
            selDist(i) = distances(tCorr(i), sCorr(i));
        end
        tCorr = tCorr(sCorr <= numSrcClusters);
        selDist = selDist(sCorr <= numSrcClusters);
        sCorr = sCorr(sCorr <= numSrcClusters);
        % Sort selected correspondences
        [selDist, idxs] = sort(selDist);
        sCorr = sCorr(idxs);
        tCorr = tCorr(idxs);
       
        % Assign background label to those which are not selected
        if(~input.includeBgClass)
            if(sum(isBgClass) == 1)
                aux_sCorr = sCorr;
                correctLabels = find(~isBgClass);
                for i = 1:length(correctLabels)
                    sCorr(aux_sCorr == i) = correctLabels(i);
                end
                if(~input.isWild)
                    allT = 1:numTgtClusters;
                    newBg = allT(~ismember(allT, tCorr))';
                    sCorr = [sCorr; find(isBgClass)*ones(length(newBg),1)];
                    tCorr = [tCorr; newBg];
                    selDist = [selDist; dist_lambdaST*ones(length(newBg),1)];
                end
                % Add fixed unknown labels
                if(input.isClassSupervised && strcmpi(input.typeWildSupervision,'100'))
                    tCorr(tCorr >= min(bgCentroids)) = tCorr(tCorr >= min(bgCentroids)) + length(bgCentroids);
                    tCorr = [tCorr; bgCentroids];
                    sCorr = [sCorr; find(isBgClass)*ones(length(bgCentroids),1)];
                end
            end
        end
        % Save all correspondences
        sCorrAll = sCorr;
        tCorrAll = tCorr;
        allDist = selDist;
        th = length(selDist) / numTgtClusters;
        
    end
    
end
function [sFeat, tFeat, W, transferIds, testFeat, classifiers] = DA_ATI(input, sData, sIds, sFeat, tData, tIds, tFeat, typeDA, testData, testIds, testFeat)
    
    global aux_feat;
    global aux_classes;
    if(input.numTgtTrainDA ~= +Inf)
        if(~input.isClassSupervised)
            aux_feat = tFeat;
            aux_classes = tData.annotations.classes;
            % aux_ids = tIds;
            perm = randperm(length(tIds));
            tData.annotations.classes = tData.annotations.classes(perm(1:min(length(tIds),input.numTgtTrainDA)));
            tIds = tIds(perm(1:min(length(tIds),input.numTgtTrainDA)));
            tFeat = tFeat(perm(1:min(length(tIds),input.numTgtTrainDA)),:);
        else
            aux_feat = testFeat;
            aux_classes = testData.annotations.classes;
            perm = randperm(length(testIds));
            testData.annotations.classes = testData.annotations.classes(perm(1:min(length(testIds),input.numTgtTrainDA)));
            testIds = testIds(perm(1:min(length(testIds),input.numTgtTrainDA)));
            testFeat = testFeat(perm(1:min(length(testIds),input.numTgtTrainDA)),:);
        end
    end
    
    numDims = size(sFeat,2);
    transferIds = [];
    classifiers = [];
    W = eye(numDims);

    % For experiment reasons, compute centroids of each label (gt: src-tgt)
    aux_SrcClusters = input.numSrcClusters;
    input.numSrcClusters = max(sIds);
    [sCentroidsGt, ~, ~, ~] = clusterSrcData(input, sData, sIds, sFeat);
    [tCentroidsGt, ~, ~, ~] = clusterSrcData(input, tData, tIds, tFeat);
    % No unknown class, so copy first shared class (results wrong -> not relevant)
    if(input.isOpenset && input.isClassSupervised)
        tCentroidsGt = [tCentroidsGt; tCentroidsGt(1,:)];
    end
    input.numSrcClusters = aux_SrcClusters;
    
    % Measurements
    outAccuracy = zeros(input.iterResetDA*input.iterDA+input.iterResetDA,2);
    angleMeanErr = zeros(input.iterResetDA*input.iterDA+input.iterResetDA,2);
    angleStdErr = zeros(input.iterResetDA*input.iterDA+input.iterResetDA,2);
    energy_gt = +Inf*ones(input.iterResetDA*input.iterDA+input.iterResetDA,1);
    energy_opt = +Inf*ones(input.iterResetDA*input.iterDA+input.iterResetDA,1);
    diff_corr = +Inf*ones(input.iterResetDA*input.iterDA+input.iterResetDA,1);

    % Train SVMs from source data and check current class. performance (non-adapted)
    if(input.isMidResultsDA)
        if(~input.isClassSupervised || input.is4ViewSupervised)
            [outAccuracy(1,:), ~, ~, angleMeanErr(1,:), angleStdErr(1,:)] = computeCurrentClassification(input, sData, sFeat, tData, tFeat, 1, 0);
        else
            sData_it = sData;
            sData_it.annotations.classes = [sData_it.annotations.classes; tData.annotations.classes];
            [outAccuracy(1,:), ~, ~, angleMeanErr(1,:), angleStdErr(1,:)] = computeCurrentClassification(input, sData_it, [sFeat; tFeat], testData, testFeat, 1, 0);
        end 
    end
%     energy_gt(1) = norm(sCentroidsGt - tCentroidsGt, 'fro');
    
    % Clustering target samples
    [tCentroids, tCenIds, tAR, tBlocks] = clusterTgtData(input, tData, tFeat);
    
    if(input.is4ViewSupervised && input.numTgtTrainDA ~= +Inf)
        final_views = [];
        for i = 1:4
            aux_view = find(tBlocks == i);
            final_views = [final_views; aux_view(1:min(length(aux_view),input.numTgtTrainDA))];
        end
        tCentroids = tCentroids(final_views,:);
        tCenIds = tCenIds(final_views);
        tBlocks = tBlocks(final_views);
    end
    
    % Nearest neightbours of target centroids (samples)
    if(~input.isClassSupervised)
        NN = getNN(tCentroids, input.numNN, input.isClosestNN);
    else
        if(input.is4ViewSupervised)
            accum = 0;
            NN = [];
            for i = 1:max(tBlocks)
                NN_aux = getNN(tCentroids(tBlocks == i,:), input.numNN, input.isClosestNN) + accum;
                accum = accum + sum(tBlocks == i);
                NN = [NN; NN_aux];
                
            end
        else
            NN = getNN([tCentroids; testFeat], input.numNN, input.isClosestNN);
        end
    end    
    
    originalSrcFeat = sFeat;
    sCentroidsRaw = clusterSrcData(input, sData, sIds, originalSrcFeat);
    sCentroidsRawGT = sCentroidsGt;
    listS = cell(size(sCentroidsGt,1),1);
    totalIt = 1;
    for itOut = 1:input.iterResetDA
    
        % -> Iterate until convergence
        for it = 1:input.iterDA
            
            totalIt = totalIt + 1;
            % Cluster source samples with average of label groups
            [sCentroids, sCenIds, sAR, sBlocks] = clusterSrcData(input, sData, sIds, sFeat);
    
            % Handling of special cases (for testing purposes)
            switch input.daSpecial
                case 'gt'
                    tCorr = (1:size(tFeat,1))';
                    selDist = zeros(length(tCorr),1);
                    [srcIds, srcClasses] = getIdLabels(input.sourceDataset, tData.annotations);
                    sCorr = srcIds;
                    % Viewpoint case (manually commented)
                    % sCorr = getAzimuthId(input.sourceDataset.azimuth, cellfun(@str2num, sCorr(:,2))); % Viewpoints
                    % Class label case
                    sCorr = getClassId(srcClasses, sCorr);
                    sCorr = sCorr(tCenIds);
                case 'rnd'
                    tCorr = (1:size(tFeat,1))';
                    sCorr = mod(randperm(length(tCorr))', input.numSrcClusters)+1;
                    selDist = rand(length(tCorr),1);
            end

            cenSrc2D = []; cenTgt2D = [];
            if(input.isDaView2D && size(tCentroids,1) == size(tFeat,1))
    %             srctgt2D = tsne([sFeat;sCentroids;tCentroids], [sIds;sBlocks;tIds]);
    %             % src2D = srctgt2D(1:length(sIds),:);
    %             cenSrc2D = srctgt2D(length(sIds)+1:length(sIds)+length(sBlocks),:);
    %             cenTgt2D = srctgt2D(length(sIds)+length(sBlocks)+1:end,:);
                srctgt2D = tsne([sCentroids;tCentroids], [sBlocks;tIds(tCenIds)]);
                cenSrc2D = srctgt2D(1:length(sBlocks),:);
                cenTgt2D = srctgt2D(length(sBlocks)+1:end,:);

            end
            % -> EVALUACION OF CLUSTERING PERFORMANCE
            if(input.isMidResultsDA)
                if(size(tCentroids,1) == size(tFeat,1))
                    f = checkClustering(it, cenSrc2D, sBlocks, cenTgt2D, tIds(tCenIds));
                end
            end

            % Assign closest points for minimisation
            if(strcmpi(input.daSpecial,''))
                if(~input.isClassSupervised || (input.isClassSupervised && ~(itOut == 1 && it == 1)) || ...
                        input.isClassSupervised && input.is4ViewSupervised)
                    [sCorr, tCorr, selDist, sCorrAll, tCorrAll, ~, th] = computeCorrespondences(input, itOut, it, NN, ...
                        sData, sFeat, sCentroids, sCenIds, sAR, sBlocks, tData, tFeat, tCentroids, tCenIds, tAR, tBlocks, ...
                        testFeat);
                else
                    sCorr = tIds;
                    sCorrAll = sCorr;
                    tCorr = (1:size(tFeat,1))';
                    tCorrAll = tCorr;
                    tCentroids = tFeat;
                    selDist = zeros(length(tCorr),1);
                    th = 1.0;
                end
            else
                sCorrAll = sCorr; tCorrAll = tCorr; th = 1;
            end

            % Histogram assignments
            list_hist = zeros(1,max(sCorr));
            for i = 1:max(sCorr)
                list_hist(i) = sum(sCorr == i);
            end
            % -> EVALUACION OF CORRESPONDENCE PERFORMANCE
            if(input.isMidResultsDA)
                if(size(tCentroids,1) == size(tFeat,1))
                    fprintf('Storing current correspondence estimation at it %d\n', it);
                    checkCorrespondences(input, f, it, itOut, cenSrc2D, sCorr, sCorrAll, sBlocks, ...
                        cenTgt2D, tCorr, tCorrAll, tIds(tCenIds), selDist, th, testIds);
                end
            end
            % Uncomment for viewpoint Refinement evaluation test
%             if(size(tCentroids,1) == size(tFeat,1))
%                 [~, srcClasses, metadata] = getIdLabels(input.sourceDataset, sData.annotations);
%                 [tgtIds, tgtClasses] = getIdLabels(input.targetDataset, tData.annotations, metadata);        
%                 strAz = srcClasses(:,ismember(metadata,'azimuth'));
%                 [~, corrIdx] = sort(tCenIds);
%                 [path, name] = getResultsPath(input);
%                 transferLabels = tgtIds;
%                 transferLabels(:,ismember(metadata,'azimuth')) = strAz(sCorr(corrIdx));
%                 confusionMatrix(input, metadata, srcClasses, tData, tgtClasses, tgtIds, transferLabels, ...
%                     [path '/' name], ['corr_it' num2str(itOut) '.' num2str(it)]);
%             end

            if(strcmpi(typeDA,'corr'))
                [~, corrIdx] = sort(tCenIds);
                transferIds = sCorr(corrIdx);
                return
            end

            % - Optimisation Process
            tCentroidsAll = double(tCentroids);
            allTgtIds = tIds;
            if(~input.is4ViewSupervised && input.isClassSupervised && ~(itOut == 1 && it == 1))
                tCentroidsAll = double([tCentroids; testFeat]);
                allTgtIds = [tIds; testIds];
            end
            
            if(it == 1 && itOut == 1 && ~input.isClassSupervised)
                energy_opt(1) = norm(sCentroids(sCorr,:) - tCentroidsAll(tCorr,:), 'fro');
                energyInit = energy_opt(1);
                energy_opt(1) = 1; 
            elseif(it == 2 && itOut == 1 && input.isClassSupervised)
                energy_opt(1) = norm(sCentroids(sCorr,:) - tCentroidsAll(tCorr,:), 'fro');
                energyInit = energy_opt(1);
                energy_opt(1) = 1;
            end
            
            % Compute W Matrix with local gradient descent Non-Linear approach
            W = eye(numDims);
            W = estimateW(input, W, numDims, double(sCentroids), sCorr, tCentroidsAll, tCorr);

            % Stop when no more different assignments from previous iter
            diffS = 0;
            for s = 1:size(sCentroids,1)
                newList = tCorr(sCorr == s);
                if(~isempty(listS{s}))
                    diffS = diffS + sum(~ismember(newList, listS{s}));
                else
                    diffS = length(tCorr);
                end
                listS{s} = newList;
            end
            fprintf('num diff: %d\n', diffS);
            if((~input.isClassSupervised && it > 1) || it > 2)
                diff_corr(totalIt) = diffS;
            end
            
            if(strcmpi(input.transformationDomain,'src'))
                d = norm((1.0 - input.deltaW)*sCentroids(sCorr,:) + input.deltaW*sCentroids(sCorr,:)*W - tCentroidsAll(tCorr,:),'fro');
            else
                d = norm((1.0 - input.deltaW)*tCentroidsAll(sCorr,:) + input.deltaW*tCentroidsAll(sCorr,:)*W - sCentroids(sCorr,:));
            end
            fprintf('Full Matrix Optimisation (it %d.%d): energy %f\n', itOut, it, d);

            if(((size(sCentroids,1) == max(allTgtIds)) && (size(tCentroidsAll,1) == length(allTgtIds))))
                sCentroidsGt = (1.0 - input.deltaW)*sCentroidsGt + input.deltaW*sCentroidsGt*W; 
                energy_gt(totalIt) = norm(sCentroidsGt - tCentroidsGt,'fro');
            end

            if(~(input.isClassSupervised && it == 1))
                energy_opt(totalIt) = d / energyInit;
            end
            if((~input.isClassSupervised && ((it > 1 && itOut == 1) || itOut > 1)) || (it > 2))
                if(abs(energy_opt(totalIt) - energy_opt(totalIt-1)) < input.tol_residual || ...
                        energy_opt(totalIt) > energy_opt(totalIt-1) || diffS == 0) % diffS since no more opt possible
                    energy_opt(totalIt) = Inf;
                    energy_gt(totalIt) = Inf;
                    sCorr = sCorrOld;
                    tCorr = tCorrOld;
                    break;
                end
            end

            if(strcmpi(input.transformationDomain,'src'))
                if(input.isFMOAllSamples)
                    if(~input.isWild)
                        % Should enter here! :)
                         sFeat = (1.0 - input.deltaW)*sFeat + input.deltaW*sFeat*W;
                    else % Only those who are non-unknown
                        isBgClass = find(ismember(input.sourceDataset.classes,'zz_unknown'));
                        transS = sFeat(sIds ~= isBgClass,:);
                        sFeat(sIds ~= isBgClass,:) = (1.0 - input.deltaW)*transS + input.deltaW*transS*W;
                    end
                else % only those who contributed in the transformation
                    deltaDyn = 1 - selDist/max(selDist);
                    deltaSrc = zeros(size(sFeat,1),1);
                    for idx = 1:max(sCenIds)
                        deltaSrc(sCenIds == idx) = single(mean(deltaDyn(sCorr == idx)));
                    end
                    deltaDynMat = repmat(deltaSrc, [1 numDims]);
                    sFeat = (1.0 - deltaDynMat).*sFeat + deltaDynMat.*(sFeat*W);
                end
            else
                if(input.isFMOAllSamples)
                    tFeat = (1.0 - input.deltaW)*tFeat + input.deltaW*tFeat*W;
                else
                    tFeat_to_transform = tFeat(tCorr,:);
                    tFeat_to_transform = (1.0 - input.deltaW)*tFeat_to_transform + input.deltaW*tFeat_to_transform*W;
                    tFeat(tCorr,:) = tFeat_to_transform;
                end
            end
            
            % Start reset after gt labels iteration (1st one)
            if(it == 1 && itOut == 1 && input.isClassSupervised && ~input.is4ViewSupervised)
                originalSrcFeat = sFeat;
            end
            % Keep old correspondences in case there is no improvement
            sCorrOld = sCorr;
            tCorrOld = tCorr;
            
            if(input.isMidResultsDA)
                if(~input.isClassSupervised || input.is4ViewSupervised)
                    [outAccuracy(totalIt,:), ~, ~, angleMeanErr(totalIt,:), angleStdErr(totalIt,:), classifiers] = computeCurrentClassification(input, sData, sFeat, tData, tFeat, itOut, it);
                else
                    sData_it = sData;
                    sData_it.annotations.classes = [sData_it.annotations.classes; tData.annotations.classes];
                    % if(it > 1)
                        [outAccuracy(totalIt,:), ~, ~, angleMeanErr(totalIt,:), angleStdErr(totalIt,:), classifiers] = computeCurrentClassification(input, sData_it, [sFeat; tFeat], testData, testFeat, itOut, it);
                    % else
                        outAccuracy(totalIt,:) = 0.5; angleMeanErr(totalIt,:) = 0.5; angleStdErr(totalIt,:) = 0.5;
                    % end
                end 
            end

        end
        
        if(it == 1 && input.iterDA ~= 1)
            break;
        end
        
        % Train SVMs from source data and check current class. performance
        if((input.iterDA > 1 && input.iterResetDA > 1) && itOut < input.iterResetDA)
            W = estimateW(input, eye(numDims), numDims, double(sCentroidsRaw), sCorr, tCentroidsAll, tCorr, selDist, ...
                connections, input.isLocalSrc, list_hist); % Src Localities
            sFeat = (1.0 - input.deltaW)*originalSrcFeat + input.deltaW*originalSrcFeat*W;
            sCentroids0 = (1.0 - input.deltaW)*sCentroidsRaw + input.deltaW*sCentroidsRaw*W;
            if(input.isMidResultsDA)
                if(~input.isClassSupervised || input.is4ViewSupervised)
                    [outAccuracy(totalIt+1,:), ~, ~, angleMeanErr(totalIt+1,:), angleStdErr(totalIt+1,:)] = ...
                        computeCurrentClassification(input, sData, sFeat, tData, tFeat, itOut+1, 0);
                else
                    sData_it = sData;
                    sData_it.annotations.classes = [sData_it.annotations.classes; tData.annotations.classes];
                    [outAccuracy(totalIt+1,:), ~, ~, angleMeanErr(totalIt+1,:), angleStdErr(totalIt+1,:)] = ...
                        computeCurrentClassification(input, sData_it, [sFeat; tFeat], testData, testFeat, itOut+1, 0);
                end
                if((size(sCentroids,1) == max(allTgtIds)) && (size(tCentroidsAll,1) == length(allTgtIds)))
                    sCentroidsGt = (1.0 - input.deltaW)*sCentroidsRawGT + input.deltaW*sCentroidsRawGT*W; 
                    energy_gt(totalIt+1) = norm(sCentroidsGt - tCentroidsGt, 'fro');
                    if(strcmpi(input.transformationDomain,'src'))
                        d = norm(sCentroids0(sCorr,:) - tCentroidsAll(tCorr,:),'fro');
                    else
                        d = norm((1.0 - input.deltaW)*tCentroidsAll(sCorr,:) + input.deltaW*tCentroidsAll(sCorr,:)*W - sCentroidsRaw(sCorr,:),'fro');
                    end
                    energy_opt(totalIt+1) = d / energyInit;
                    energy_opt(totalIt) = d;
                end
            end
            totalIt = totalIt + 1;
        end

    end
    
    path = getResultsPath(input);
    
    if(totalIt > 3 && size(energy_gt,1) > 1)
        % PLOT: energies of minimisation per iteration
        energy_gt(energy_gt == +Inf) = [];
        if(~isempty(energy_gt))
            fenergy = figure;
            set(fenergy , 'visible', 'off');
            % energy_gt = (energy_gt ./ energy_gt(1) - 1.0) * 100.0;        
            % plot(0:size(energy_gt,1)-1, zeros(size(energy_gt,1),1),'Color','r');
            hold on;
            plot(0:size(energy_gt,1)-1, energy_gt, '-x', 'MarkerSize', 10, 'MarkerFaceColor', 'b', 'LineWidth', 1.75);
            ylabel('% src-tgt gt distance change w.r.t. non adapted src');
            xlabel('Number of optimisation (FMO) iterations');
            title('Distance reduction between ground truth correspondences per iteration', 'FontSize', 12);
            % axis([0 size(energy,1)-1 0 max(energy)+1]);
    %         range = 50;
    %         if(abs(min(energy_gt)) > 75)
    %             range = 100;
    %         elseif(abs(min(energy_gt)) > 50)
    %             range = 75;
    %         end
            axis([0 size(energy_gt,1)-1 energy_gt(1) - energy_gt(1)/2 energy_gt(1) + energy_gt(1)/2]);
            ax = gca;
            set(ax,'XTick',0:size(energy_gt,1)-1);
            legend('non-adapted reference','distance gt src <> tgt','classification accuracy','Location','northeast');
            saveas(fenergy, [path '/progressEnergyGT.pdf']);
            saveas(fenergy, [path '/progressEnergyGT.png']);
        end

        % PLOT: Number of different correspondences w.r.t. last iteration
        energy_opt(energy_opt == +Inf) = [];
        if(~isempty(energy_opt))
            fenergy = figure;
            set(fenergy , 'visible', 'off');
            plot(0:size(energy_opt,1)-1, zeros(size(energy_opt,1),1),'Color','r');
            hold on;
            energy_opt = (energy_opt ./ energy_opt(1) - 1.0) * 100.0;        
            plot(0:size(energy_opt,1)-1, energy_opt, '-x', 'MarkerSize', 10, 'MarkerFaceColor', 'b', 'LineWidth', 1.75);
            ylabel('% src-tgt gt distance change w.r.t. non adapted src');
            xlabel('Number of optimisation (FMO) iterations');
            title('Distance between estimated correspondences per iteration', 'FontSize', 12);
            range = 50;
            if(abs(min(energy_opt)) > 75)
                range = 100;
            elseif(abs(min(energy_opt)) > 50)
                range = 75;
            end
            axis([0 size(energy_opt,1)-1 -range range]);
            ax = gca;
            set(ax,'XTick',0:size(energy_opt,1)-1);
            legend('non-adapted reference','distance gt src <> tgt','classification accuracy','Location','southeast');
            saveas(fenergy, [path '/progressEnergyOpt.pdf']);
            saveas(fenergy, [path '/progressEnergyOpt.png']);
        end

        % PLOT: Number of different correspondences w.r.t. last iteration
        diff_corr(diff_corr == +Inf) = [];
        if(length(diff_corr) > 1)
            if(max(diff_corr) ~= 0)
                fenergy = figure;
                set(fenergy , 'visible', 'off');
                hold on;
                extraIt = 1;
                if(input.isClassSupervised)
                    extraIt = 2;
                end
                plot(1+extraIt:size(diff_corr,1)+extraIt, diff_corr, '-x', 'MarkerSize', 10, 'MarkerFaceColor', 'b', 'LineWidth', 1.75);
                ylabel('Number different correspondences src-tgt (current vs. previous iteration)');
                xlabel('Number of optimisation (FMO) iterations');
                title('Number of different correspondences w.r.t. last iteration', 'FontSize', 12);
                axis([1+extraIt size(diff_corr,1)+extraIt 0 max(diff_corr)]);
                ax = gca;
                set(ax,'XTick',1:size(diff_corr,1)+extraIt);
                saveas(fenergy, [path '/diffCorr.pdf']);
                saveas(fenergy, [path '/diffCorr.png']);
            end
        end

        if(input.isMidResultsDA)
            % Print progress of classification performance
            numZero = (outAccuracy == 0);
            outAccuracy(numZero) = [];
            if(sum(numZero) > 0)
                outAccuracy = reshape(outAccuracy,length(outAccuracy)/2,2);
            end
            fend = figure;
            set(fend, 'visible', 'off');
            plot(0:size(outAccuracy,1)-1, outAccuracy(1,2)*ones(size(outAccuracy,1),1),'Color','r');
            hold on;
            plot(0:size(outAccuracy,1)-1, outAccuracy(:,2), '-x', 'MarkerSize', 10, 'MarkerFaceColor', [0.7,0.7,0.7], 'Color', [0.7,0.7,0.7], 'LineWidth', 1.75); % , 'LineSmoothing', 'on');
            ylabel('Classication accuracy (%)');
            xlabel('Number of DA iterations');
            title('Progress of classification performance (mean probabilities)', 'FontSize', 12);
            axis([0 size(outAccuracy,1)-1 30 100]);
            ax = gca;
            set(ax,'XTick',0:size(outAccuracy,1)-1);
            legend('without adaptation','with adaptation','Location','southeast');
            saveas(fend, [path '/progressClassification (prob).pdf']);
            hold off;

            % Print progress in angle mean and std if viewpoints available
            if(isprop(input.sourceDataset,'azimuth'))
                plot(0:size(angleMeanErr,1)-1, angleMeanErr(1,1)*ones(size(angleMeanErr,1),1),'Color', [0.2,0.3,0.8],'LineWidth', 0.9);
                hold on;
                plot(0:size(angleMeanErr,1)-1, angleMeanErr(:,1), '-x', 'MarkerSize', 10, 'MarkerFaceColor', [0.2,0.3,0.8], 'Color', [0.2,0.3,0.8], 'LineWidth', 1.75); %, 'LineSmoothing', 'on');
                plot(0:size(angleStdErr,1)-1, angleStdErr(1,1)*ones(size(angleStdErr,1),1),'Color', [0.5,0.6,1.0],'LineWidth', 0.9);
                plot(0:size(angleStdErr,1)-1, angleStdErr(:,1), '-x', 'MarkerSize', 10, 'MarkerFaceColor', [0.5,0.6,1.0], 'Color', [0.5,0.6,1.0], 'LineWidth', 1.75); %, 'LineSmoothing', 'on');
                legend('mean','std dev','Location','northeast');
                hold off;
                ylabel('Angle error in degrees');
                xlabel('Number of DA iterations');
                title('Mean and StdDev Angle error (degrees)', 'FontSize', 12);
                range = 90;
                if(max(angleStdErr) < 60)
                    range = 60;
                end
                if(input.is4ViewSupervised)
                    range = 45;
                    if(max(angleStdErr) < 10)
                        range = 10;
                    elseif(max(angleStdErr) < 25)
                        range = 25;
                    end
                end
                axis([0 size(outAccuracy,1)-1 0 range]);
                ax = gca;
                set(ax,'XTick',0:size(outAccuracy,1)-1);
                legend('Mean w/o adaptation','Mean with adaptation','StdDev w/o adaptation','StdDev with adaptation','Location','northeast');
                saveas(fend, [path '/angle error.pdf']);
            end
            % leaveMatlab();
        end
    end

    if(input.numTgtTrainDA ~= +Inf)
        if(~input.isClassSupervised)
            tFeat = aux_feat;
            tData.annotations.classes = aux_classes;
        else
            testFeat = aux_feat;
            testData.annotations.classes = aux_classes;
        end
    end
end

function refinementSVM(input, transferLabels, scores, srcFeatures, srcData, srcCentres, srcAll, tgtFeatures, tgtData, tgtCentres, tgtLabels, distancesAll, th, thAddSVM, id)

    [~, tgtClasses] = getIdLabels(input.targetDataset, tgtData.annotations);
    aux_labels = zeros(size(transferLabels,1),1);
    for i = 1:size(tgtClasses,1)
        idClass = ismember(transferLabels,tgtClasses(i,:));
        values = sum(idClass,2);
        aux_labels(max(values) == values) = i;
    end
    strTransferLabels = transferLabels;
    transferLabels = aux_labels;

    if(th == 1)
        srcIds = srcAll(distancesAll <= mean(distancesAll));
    else
        srcIds = srcAll(distancesAll <= max(distancesAll));
    end
    maxAssigns = floor(size(tgtCentres,1) / size(srcCentres,1));
    probAssigns = zeros(size(srcCentres,1),1);
    for i = 1:size(srcCentres,1)
        probAssigns(i) = sum(srcIds == i) / maxAssigns;
    end
    srcCandidates = find(probAssigns > thAddSVM);
    candSamples = [];
    for i = 1:length(srcCandidates)
        candSamples = [candSamples; find(transferLabels == srcCandidates(i))];
    end
    toAdd = [];
    for i = 1:length(candSamples)
        s = scores(candSamples(i),:);
%         idxs = (s - median(s)) > 1.48*median(abs(s - median(s)));
        s = s - min(s);
        % Estimate MAD
        M = median(s);
        sM = s - M;
        idxs = find(sM > 1.4826*M);
        if(length(idxs) == 1)
            toAdd = [toAdd; candSamples(i)];
        end
    end    
    
    path = getResultsPath(input);
    
    if(~isempty(toAdd))
        % Calculate home any right choices!
        matches = sum(tgtLabels(toAdd) == transferLabels(toAdd));
        acc = matches / length(toAdd);
        strAcc = sprintf('%.2f',acc);
        fmatch = figure;
        set(fmatch, 'visible', 'off');
        saveas(fmatch, [path '/ref.s-t' num2str(id) '_' num2str(th) '_' num2str(thAddSVM) '_' strAcc '(' num2str(matches) '.' num2str(length(toAdd)) ').png']);
        % Retrain with new samples from the training
        newFeatures = [srcFeatures; tgtFeatures(toAdd,:)];
        addition = strTransferLabels(toAdd,:);
        srcData.annotations.classes = [srcData.annotations.classes; addition(:,1)];
        if(isfield(srcData.annotations,'azimuth'))
            nums = cellfun(@str2num,addition(:,2));
            srcData.annotations.azimuth = [srcData.annotations.azimuth; nums];
        end
        computeCurrentClassification(input, srcData, newFeatures, tgtData, tgtFeatures, id+1);
    end

end

function [tgtCentres, tgtCentresLabels, estLabels] = clusterSVM(input, srcData, srcFeatures, tgtData, tgtFeatures)

    % Train SVMs from source data
    [w_pos, w_neg, bias] = trainLabels(input, srcData, srcFeatures);
    % Assign cluster to target data using SVMs
    transferLabels = assignLabels(input, tgtData, tgtFeatures, w_pos, w_neg, bias);
    
    % Estimate centroids for each cluster
    id = getIdLabels(input, srcData.labels);
    estLabels = getIdLabels(input, transferLabels);
    tgtCentres = zeros(max(id), size(tgtFeatures,2));
    tgtCentresLabels = zeros(max(id), 1);
    for label = 1:max(id)
        tgtCentres(label,:) = mean(tgtFeatures(estLabels == label,:));
        tgtCentresLabels(label) = label;
    end

end

function [res, transferLabels, scores, angleMeanErr, angleStdErr, classifiers] = computeCurrentClassification(input, srcData, srcFeatures, tgtData, tgtFeatures, itOut, it)

    global aux_feat;
    global aux_classes;
    if(input.numTgtTrainDA ~= +Inf)
        tgtFeatures = aux_feat;
        tgtData.annotations.classes = aux_classes;
    end

    fprintf('Storing current classification it %d.%d\n', itOut,it);
    isStorage = false; isVerbose = false;
    [path, name] = getResultsPath(input);
    path = [path '/' name];
    [srcIds, srcClasses, metadata] = getIdLabels(input.sourceDataset, srcData.annotations);
    [tgtIds, tgtClasses] = getIdLabels(input.targetDataset, tgtData.annotations, metadata);        
    classifiers = trainLabels(input, srcData, srcFeatures, srcIds, srcClasses, input.typeClassifier, isStorage, isVerbose);

    isSup = input.isClassSupervised;
    input.isClassSupervised = false;
    % Assign cluster to target data using SVMs
    [transferLabels, scores] = assignLabels(input, classifiers, srcClasses, metadata, tgtIds, tgtFeatures, isVerbose);
    if(input.isWSVM)
        isBg = cellfun(@isempty,transferLabels);
        transferLabels(isBg) = repmat({'zz_unknown'}, [sum(isBg) 1]);
    end
    input.isClassSupervised = isSup;

    if(~strcmpi(input.methodSVM, 'liblinear')) % lsvm DA type
        % Save intermediate results
        [res, angleMeanErr, angleStdErr] = confusionMatrix(input, metadata, srcClasses, tgtData, tgtClasses, tgtIds, transferLabels, path , ['it' num2str(itOut) '.' num2str(it)]);
        res = res(1,:);
        angleMeanErr = angleMeanErr(1,:);
        angleStdErr = angleStdErr(1,:);
    else
        res = 0; angleMeanErr = 0; angleStdErr = 0;
    end

    if(input.numTgtTrainDA ~= +Inf)
        aux_feat = tgtFeatures;
        aux_classes = tgtData.annotations.classes;
    end
    
end

function [src2D, cenSrc2D, tgt2D, cenTgt2D] = get2DVis(srcCentres, srcCentresLabels, srcFeatures, srcLabels, ...
    tgtCentres, tgtCentresLabels, tgtFeatures, tgtLabels)
    
    vis2D = tsne([srcCentres; srcFeatures; tgtCentres; tgtFeatures], ...
        [srcCentresLabels; srcLabels; tgtCentresLabels; tgtLabels]);
    cenSrc2D = vis2D(1:length(srcCentresLabels),:);
    src2D = vis2D(length(srcCentresLabels)+1:length(srcCentresLabels)+length(srcLabels),:);
    cenTgt2D = vis2D(length(srcCentresLabels)+length(srcLabels)+1:...
        length(srcCentresLabels)+length(srcLabels)+length(tgtCentresLabels),:);
    tgt2D = vis2D(length(srcCentresLabels)+length(srcLabels)+length(tgtCentresLabels)+1:end,:);

end

function f = checkClustering(it, srcCen, sBlocks, tgtCen, gtLabels)

    f = figure;
    set(f, 'visible', 'off');
    set(f, 'Position', [500, 100, 1024, 1024]);
    hold on;
    if(~isempty(srcCen))
        rng(10);
        color = rand([max(sBlocks),3]);
        for c = 1:max(gtLabels)
            scatter(tgtCen(gtLabels == c,1), tgtCen(gtLabels == c,2), 50, color(c,:), 'filled', 'o');
        end
        if(~isempty(srcCen))
            strS = 'S';
            for c = 1:max(sBlocks)
                % scatter(src2D(sIds == c,1), src2D(sIds == c,2), 50, color(c,:), 'x');
                text(srcCen(sBlocks == c,1),srcCen(sBlocks == c,2), strS, 'Color', color(c,:), 'FontSize', 16);
            end
            % Define boundaries
            bMin = min([srcCen; tgtCen]);
            bMax = max([srcCen; tgtCen]);
            axis([bMin(1)-abs(bMin(1)*0.1) bMax(1)+abs(bMax(1)*0.1) bMin(2)-abs(bMin(2)*0.1) bMax(2)+abs(bMax(2)*0.1)]);
        %     strAcc = sprintf('[it: %d] cluster accuracy: %.2f%', it, clusterAccuracy);
            strAcc = sprintf('[it: %d]', it);
            title(strAcc, 'FontSize', 12);
        end
    end
    hold off;
    
end

function checkCorrespondences(input, f, it, itOut, cenSrc2D, sCorr, sCorrAll, sBlocks, ...
    tgt2D, tCorr, tCorrAll, tIds, distances, th, testIds)

    figure(f);
    set(f, 'visible', 'off');
    hold on;
    
    if(~input.is4ViewSupervised && input.isClassSupervised && ~(itOut == 1 && it == 1))
        tIds = [tIds; testIds];
    end
    
    % Accuracy in correspondence computing (+ Precision + Recall)
    p = zeros(length(tCorr),1);
    r = zeros(length(tCorr),1);
    for idxCorr = 1:length(tCorr)
        
        if(~isempty(cenSrc2D))
            p1 = cenSrc2D(sCorr(idxCorr),:);
            p2 = tgt2D(tCorr(idxCorr),:);
        end
        
        tgtLabelSamples = tIds(tCorr(idxCorr));
        srcLabel = sBlocks(sCorr(idxCorr));
        p(idxCorr) = sum(tgtLabelSamples == srcLabel);
        r(idxCorr) = sum(tgtLabelSamples == srcLabel);

        if(~isempty(cenSrc2D))
            if(srcLabel(1) == tgtLabelSamples(1))
                colour = [0.1 0.1 0.8];
            else
                colour = [0.8 0.1 0.1];
            end
            scatter(tgt2D(tCorr(idxCorr),1), tgt2D(tCorr(idxCorr),2), 70, colour, 'o', 'LineWidth', 1);
            line([p1(1) p2(1)], [p1(2) p2(2)], 'Color', colour, 'LineWidth', 2); % , 'LineSmoothing', 'on');
        end
        
    end
    
    rTotal = sum(r) / length(tIds);
    pTotal = sum(p) / length(tCorr);

    if(~isempty(cenSrc2D))
        h = get(gca, 'Title');
        oldTitle = get(h,'String');
        newTitle = sprintf(' correspon. prec=%.1f rec=%.1f with lambda at %.2f', pTotal*100, rTotal*100, input.numLambda);
        newTitle = strcat(oldTitle, newTitle);
        title(newTitle, 'FontSize', 12);
        axis off;
        hold off;
        path = getResultsPath(input);
        prec = sprintf('%.1f',pTotal*100);
        rec = sprintf('%.1f',rTotal*100);
        saveas(f, [path '/corresp_' num2str(itOut) '.' num2str(it) '_p' prec '_r' rec '.pdf']);
    end

    % -> Draw PR of distances
    % Accuracy in correspondence computing (+ Precision + Recall)
    % RECALL = good corrs / all gt corrs
    % PRECISION = good corrs / all selected corrs
    [~, idxs] = sort(distances); %, 'descend');
    tCorr = tCorr(idxs);
    sCorr = sCorr(idxs);
    
    p = zeros(length(tCorr),1);
    r = zeros(length(tCorr),1);
    numCorrect = 0;
    for idxCorr = 1:length(tCorr)
        tgtLabelSamples = tIds(tCorr(idxCorr));
        srcLabel = sBlocks(sCorr(idxCorr));
        if(tgtLabelSamples == srcLabel)
            numCorrect = numCorrect + 1;
        end
        r(idxCorr) = numCorrect / length(tIds);
        p(idxCorr) = numCorrect / idxCorr;
    end
    AP = VOCap(r, p);
    f1 = figure;
    set(f1, 'visible', 'off');
    set(f1, 'Position', [500, 100, 1024, 512]);
    hold on;
    plot(r, p, 'LineWidth', 2, 'Color', [0,0,0]); % , 'LineSmoothing', 'on');
    range = [-0.01 1.01]; xlim(range); ylim(range);
    xlabel('Recall');
    ylabel('Precision');
    srcTitle = sprintf('[it: %d] Precision / Recall of src-tgt correspondences: pr-rcll [%.1f,%.1f] AP: %.1f', it, p(end)*100, r(end)*100, AP*100);
    title(srcTitle, 'FontSize', 12);
    hold off;
    prec = sprintf('%.1f',p(end)*100); rec = sprintf('%.1f',r(end)*100); strAP = sprintf('%.1f', AP*100);
    path = getResultsPath(input);
    saveas(f1, [path '/correspPR_' num2str(itOut) '.' num2str(it) '_p' prec '_r' rec '_AP' strAP '.pdf']);
    saveas(f1, [path '/correspPR_' num2str(itOut) '.' num2str(it) '_p' prec '_r' rec '_AP' strAP '.png']);
    
end

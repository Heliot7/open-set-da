% ->  Types transform: (1) "translation", (2) "scaling", (3) "transScale" translation+scaling
function sFeat = DA_transScale(typeTransform, input, sData, sIds, sFeat, tData, tFeat)

    % Clustering of target samples
    [tCentroids, tCenIds, tAR, tBlocks] = clusterTgtData(input, tData, tFeat);
    % Loop Variables
    numDims = size(sFeat,2);
    numIter = 3;
    th = 0.0001;
    residual_old = Inf;
    % Iterate until max num of iterations or convergence
    for it = 1:numIter
        % Cluster source samples with average of label groups
        [sCentroids, sCenIds, sAR, sBlocks] = clusterSrcData(input, sData, sIds, sFeat);
        % Assign closest points for minimisation
        [sCorr, tCorr] = computeCorrespondences(input, it, sData, sFeat, sCentroids, sCenIds, sAR, sBlocks, tData, tFeat, tCentroids, tCenIds, tAR, tBlocks);
        % (1) Compute linear transformation
        % (2) Update current iteration results (for residual check)
        switch input.typeDA
            case 'translation'
                translations = solveTranslation(sCentroids(sCorr,:), tCentroids(tCorr,:));
                scales = ones(1,length(translations));
            case 'scale'
                translations = zeros(1,length(scales));
                scales = solveScale(sCentroids(sCorr,:), tCentroids(tCorr,:));
            case 'transScale'
                [translations, scales] = solveTransScale(sCentroids(sCorr,:), tCentroids(tCorr,:));                
        end
        transMat = sparse(1:numDims,1:numDims, scales);
        transMat = [[transMat, translations']; [zeros(1,numDims), 1]];
        residual = norm(full(transMat*sparse([double(sCentroids(sCorr,:))'; ones(1,length(sCorr))])) - [tCentroids(tCorr,:)'; ones(1,length(sCorr))]);
        if(abs(residual_old - residual) < th)
            fprintf('[it %d] No further improvements... exit minimisation loop\n', it);
            break;
        end
        residual_old = residual;           
        fprintf('[it %d] Linear transformation - %s: energy %.3f\n', it, typeTransform, residual);
        % Update features after transformatin
        sFeat = sFeat .* repmat(scales,[size(sFeat,1) 1]) + repmat(translations,[size(sFeat,1) 1]);
    end
    
end

function newLabels = transferKNN(srcFeatures, idLabels, strLabels, tgtData, tgtFeatures, numDim)

    if(nargin < 7)
        numDim = round(0.10*size(tgtFeatures,2));
    end

    numRealSamples = length(tgtData.annotations.imgId);

    % PCA reduction to improve performance and reduce storage in memory
    fprintf('k-Nearst Neighbour algorithm (PCA - %d dims)\n', numDim);  
    numKK = round(sqrt(size(tgtFeatures,2)));
    P = mPCA([tgtFeatures; srcFeatures]);
    tgtFeatures = tgtFeatures * P(:,1:numDim);
    srcFeatures = srcFeatures * P(:,1:numDim);
    
    labels = zeros(numRealSamples,1);
    kdTree = kdtree_build(double(srcFeatures));
    for i = 1:numRealSamples
        
        fprintf('Assigning label for sample %d\n', i);
        closestCands = kdtree_k_nearest_neighbors(kdTree, double(tgtFeatures(i,:)), numKK);
         
         labelCandidates = zeros(length(strLabels),1);
         labels(i) = 1;
         for j = 1:length(closestCands)

             idx = find(prod(ismember(strLabels,idLabels(closestCands(j),:)),2));
             labelCandidates(idx) = labelCandidates(idx) + 1;
             [~, maxLabel] = max(labelCandidates);
             if(idx == maxLabel)
                 labels(i) = idx;
             end
         end
        
    end

    kdtree_delete(kdTree);
    
    newLabels = cell(numRealSamples,size(strLabels,2));
    for idxLabel = 1:length(strLabels)
        isLabel = (labels == idxLabel);
        if(~isempty(isLabel))
            newLabels(isLabel,:) = repmat(strLabels(idxLabel,:),[sum(isLabel) 1]);
        end
    end
    
end


function [centroids, labels] = clusterSamples(features, numClusters, ar, numIter)

    if(nargin < 5)
        numIter = 10;
    end  

    if(size(features,1) <= numClusters)
        centroids = features;
        labels = (1:size(features,1))';
    else
        minDims = min(size(features,2), 1024); % 1024 hardcoded
        [coeff, featuresPCA] = mPCA(features);
        if(size(featuresPCA,2) > minDims)
            featuresPCA = featuresPCA(:,1:minDims);        
        end
        t = tic;
        [labels, centroids] = kMeansPlusPlus(featuresPCA', numClusters, numIter);
%         [centroids, labels] = mexKMeans(double(featuresPCA), ar, numClusters, numIter);
        fprintf('K-Means in %s sec\n', sprintf('%.2f',toc(t)));
        if(size(coeff,2) > minDims)
            centroids = centroids * coeff(:,1:minDims)';
        end
    end

end

% - TODO: K-Means Mahalanobis distances
% [centroids, covs, labels] = KMeans_Mahalanobis(featuresPCA,numClusters, numIter);
% tgtCov = zeros(size(centroids,1),size(centroids,2),size(centroids,2));
% for c = 1:size(centroids,1)
%     samples = featuresPCA(labels == c,:);
%     tgtCov(c,:,:) = cov(samples);
%     ep = squeeze(tgtCov(c,:,:));
%     invEp2 = inv(ep);
%     detEp2 = det(ep);
% end

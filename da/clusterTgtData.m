function [centroids, labels, arCentroids, supBlocks] = clusterTgtData(input, data, feat)

    numTgtClusters = min(input.numTgtClusters, size(feat,1));
    if(isfield(data.annotations, 'BB'))
        arSamples = data.annotations.BB(:,3) ./ data.annotations.BB(:,4);
    else
        arSamples = zeros(size(feat,1),1);
    end
    if(~input.isClassSupervised)
        % Clustering adding AR of samples
        [centroids, labels] = clusterSamples(feat, numTgtClusters, arSamples);
        % Compute AR of clusters
        arCentroids = zeros(numTgtClusters, 1);
        if(isfield(data.annotations, 'BB'))
            for i = 1:numTgtClusters
                bbCluster = data.annotations.BB(labels == i,:);
                arCentroids(i) = mean(bbCluster(:,3)) ./ mean(bbCluster(:,4));
            end
        end
        % What supervised block does a sample belong to?
        % -> unsupervised = 1, 4 view = 4, supervised = #fine
        supBlocks = ones(numTgtClusters,1);
    else
        % Separate all classes
        classes = input.targetDataset.classes;
        centroids = []; arCentroids = []; supBlocks = [];
        labels = zeros(size(feat,1),1);
%         labels = [];
        for c = 1:length(classes)
            idClass = ismember(data.annotations.classes,classes{c});
            % -> 4 View Samples, BB and AR
            if(input.is4ViewSupervised)
                lowerBound = [315, 45, 135, 225];
                upperBound = [45, 135, 225, 315];                
                angles = data.annotations.vp.azimuth;
                viewClusters = [];
                stepSize = (360/length(input.sourceDataset.azimuth))/2;
                listViewpoints = input.sourceDataset.azimuth;
                if(length(listViewpoints) == 36)
                    stepSize = 0;
                end       
                for i = 1:4 % front, rear, left, right views
                    if(i == 1) % front view -> special case
                        samples = angles > lowerBound(i) + stepSize | angles < upperBound(i) - stepSize;
                        minCandidates = length(find(listViewpoints > lowerBound(1) | listViewpoints < upperBound(1)));
                    elseif(i == 3)
                        samples = angles > lowerBound(i) + stepSize & angles < upperBound(i) - stepSize;
                        minCandidates = length(find(listViewpoints > lowerBound(3) & listViewpoints < upperBound(3)));
                    else
                        samples = angles >= lowerBound(i) - stepSize & angles <= upperBound(i) + stepSize;
                        minCandidates = length(find(listViewpoints >= lowerBound(i) & listViewpoints <= upperBound(i)));
                    end
                    propViews = length(listViewpoints)/minCandidates;
                    [centroids, labels, arCentroids, numSubClusters] = ...
                        calculateCentroids(centroids, labels, arCentroids, numTgtClusters, ...
                        samples & idClass, data.annotations, feat, arSamples, max(minCandidates, round(numTgtClusters/propViews)));
                    viewClusters = [viewClusters; ((c-1)*4+i)*ones(numSubClusters,1)];
                end
                supBlocks = [supBlocks; viewClusters];
            else % -> No viewpoints known but class ids
                
                [centroids, labels, arCentroids, numSubClusters] = ...
                        calculateCentroids(centroids, labels, arCentroids, numTgtClusters, ...
                        idClass, data.annotations, feat, arSamples, round(numTgtClusters/length(classes)));
                supBlocks = [supBlocks; c*ones(numSubClusters,1)];
            end
        end
    end

end

function [centroids, labels, arCentroids, numSubClasses] = calculateCentroids(centroids, labels, arCentroids, numTgtClusters, samples, annot, feat, arSamples, minSubClasses)

    % Although we tried to specify the same amount...
    % ... some classes cannot provide more samples
    if(numTgtClusters == size(feat,1))
        numSubClasses = sum(samples);
    else
        numSubClasses = min(sum(samples), minSubClasses);
    end
    % Clustering adding AR of samples
    [tgtCluster, tgtId] = clusterSamples(feat(samples,:), numSubClasses, arSamples(samples));
    centroids = [centroids; tgtCluster];
    listSamples = find(samples);
    if(length(listSamples) == numSubClasses)
        labels(length(find(labels))+tgtId) = listSamples;
    else
        labels(samples) = max(labels)+tgtId;
    end

    % Compute AR of clusters
    if(isfield(annot, 'BB'))
        bb = annot.BB(samples,:);
        for j = 1:numSubClasses
            bbCluster = bb(tgtId == j,:);
            arCentroids = [arCentroids; mean(bbCluster(:,3)) ./ mean(bbCluster(:,4))];
        end
    else
        arCentroids = [arCentroids; zeros(numSubClasses,1)];
    end

end

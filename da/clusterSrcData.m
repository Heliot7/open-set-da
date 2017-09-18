function [centroids, labels, arCentres, supBlocks] = clusterSrcData(input, data, labels, feat)
    
    % 1 Cluster = 1 Label
    if(input.numSrcClusters <= max(labels))
        numSrcClusters = max(labels);
        centroids = zeros(numSrcClusters, size(feat,2), 'single');
        arCentres = zeros(numSrcClusters, 1);
        for i = 1:numSrcClusters
            samples = feat(labels == i,:);
            centroids(i,:) = mean(samples);
            if(isfield(data.annotations, 'BB'))
                bbCluster = data.annotations.BB(labels == i,:);
                arCentres(i) = mean(bbCluster(:,3)) ./ mean(bbCluster(:,4));
            end
        end
        supBlocks = (1:numSrcClusters)';
    % 1 Cluster = 1 Sample        
    elseif(input.numSrcClusters >= size(feat,1))
        numSrcClusters = size(feat,1);
        centroids = feat;
        if(isfield(data.annotations, 'BB'))
            arCentres = data.annotations.BB(:,3) ./ data.annotations.BB(:,4);
        else
            arCentres = zeros(numSrcClusters, 1);
        end
        supBlocks = labels;
    % 1 Label > 1 Cluster > 1 Sample
    else
        centroids = []; supBlocks = []; newLabels = []; 
        innerClusters = floor(input.numSrcClusters / max(labels));
        arCentres = zeros(innerClusters*max(labels), 1);
        for idxLabel = 1:max(labels)
            samples = feat(labels == idxLabel,:);
            arSamples = zeros(size(samples,1),1);
            [aux_centres, aux_labels] = clusterSamples(samples, innerClusters, arSamples);
            centroids = [centroids; aux_centres];
            newLabels = [newLabels; innerClusters*(idxLabel-1)+aux_labels];
            supBlocks = [supBlocks; idxLabel*ones(innerClusters,1)];
        end
        % Compute AR of clusters
        if(isfield(data.annotations, 'BB'))
            for i = 1:length(arCentres)
                bbCluster = data.annotations.BB(newLabels == i,:);
                arCentres(i) = mean(bbCluster(:,3)) ./ mean(bbCluster(:,4));
            end
        end
        labels = newLabels;
    end

end
function dist = medianDist(sFeat, srcSamplesCluster, tFeat, tgtSamplesCluster)

    dist = zeros(max(tgtSamplesCluster), max(srcSamplesCluster), 'single');
    for i = 1:max(tgtSamplesCluster)
        
        tgtSamples = tFeat(tgtSamplesCluster == i,:);
        for j = 1:max(srcSamplesCluster)
            
            srcSamples = sFeat(srcSamplesCluster == j,:);
            L1 = zeros(size(tgtSamples,1)*size(srcSamples,1),size(srcSamples,2));
            idx = 1;
            for s = 1:size(tgtSamples,1)
                L1(idx:idx-1+size(srcSamples,1),:) = repmat(tgtSamples(s,:), [size(srcSamples,1) 1]) - srcSamples;
                idx = idx + size(srcSamples,1);
            end
            m = median(L1);
            dist(i,j) = sqrt(sum(m.*m, 2));
            
        end
        
    end

end
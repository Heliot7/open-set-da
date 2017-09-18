function nn = getNN(points, K, isClosestNN)

    fprintf('Estimating NNs of target samples\n');
    numElems = size(points,1);
    if(nargin < 2)
        K = numElems-1;
    end

    % (1) compute ALL distances and closest NN points
    nn = -1*ones(numElems,K);
    nnDist = -1*ones(numElems,K);
    if(~isempty(nn))
        points = single(points);
        for i = 1:numElems
            L1 = repmat(points(i,:),[numElems 1]) - points;
            [dist, idx] = sort(sum(L1.*L1, 2));
            nn(i,:) = idx(2:1+K);
            nnDist(i,:) = dist(2:1+K);
        end
    end
    
    if(isClosestNN)
        % Median of all distances
        % thDist = median(distances(:));
        % minDist = min(distances(:)); maxDist = max(distances(:));
        % thDist = 0.1*(maxDist - minDist) + minDist;
        thDist = median(nnDist(:));
        % Take those with lower distances (median) that satisfy for the K's NN
        isOk = nnDist < thDist;
        % nn(sum(isOk,2) < K,:) = 0;
        nn(~isOk) = 0; % dynamic (NN = n might have NN < n included)
    end

end

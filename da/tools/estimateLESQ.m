function [W, selectedDims] = estimateLESQ(input, W, numIter, numDims, sCentroids, sCorr, sFeat, sIds, tCentroids, tCorr)

    minOk = numDims; %floor(sqrt(numDims));

    X = sCentroids(sCorr,:);
    Y = tCentroids(tCorr,:);

    vpStd = zeros(size(sCentroids,1),numDims);
    for vp = 1:size(sCentroids,1)
        vpFeat = sFeat(sIds == vp,:);
        vpStd(vp,:) = std(vpFeat);
    end
    meanAllStd = median(vpStd);
    [~, idxs] = sort(meanAllStd);
    selectedDims = idxs(1:minOk);
    subX = X(:,selectedDims);
    subY = Y(:,selectedDims);

%     W = (subX' * subX) \ (subX' * subY);
    W = X \ Y;

%     numCorr = length(sCorr);
%     A = zeros(numCorr, minOk, 'single');
%     b = subY(:); % b = zeros(minOk*minOk,1);
%     
%     colA = 1;
%     for i = 1:minOk
%         for j = 1:minOk
%             idxs = sub2ind([numCorr,minOk],1:numCorr,j*ones(1,numCorr));
%             A(idxs,colA) = subX(:,i);
%             % b(rowA) = subY(i,j);
%             colA = colA + 1;
%         end
%         fprintf('numDims done %d/%d\n',i,minOk);
%     end
%     % W = A \ b;
%     W = (A' * A) \ A' * b;
%     W = reshape(W, minOk, minOk);
end

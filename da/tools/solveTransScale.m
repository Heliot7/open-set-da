function [translations, scales] = solveTransScale(src, tgt)

    numConnections = size(src,1);
    numDims = size(src,2);

    b = tgt'; b = double(b(:));
    At = repmat(sparse(1:numDims,1:numDims,1),[numConnections 1]);
    As = sparse(numDims*numConnections,numDims);
    for idxElem = 0:numConnections-1
        As(1+idxElem*numDims:idxElem*numDims+numDims,:) = diag(src(idxElem+1,:));
    end
    A = [As, At];
    cvx_begin quiet
        variable x(2*numDims) % scale and translation (twice dims)
        minimize( norm( A*x - b ) )
    cvx_end
    scales = x(1:length(x)/2)'; translations = x(length(x)/2+1:end)';

end
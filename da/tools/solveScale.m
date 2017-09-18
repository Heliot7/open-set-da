function scales = solveScale(src, tgt)

    numConnections = size(src,1);
    numDims = size(src,2);

    b = tgt'; b = b(:);
    A = sparse(numDims*numConnections,numDims);
    for idxElem = 0:numConnections-1
        A(1+idxElem*numDims:idxElem*numDims+numDims,:) = diag(src(idxElem+1,:));
    end
    x = directSolutionSystemLinearEquations(A, b(:));
%     cvx_begin quiet
%         variable x(numDims) % scale
%         minimize( norm( A*x - b ) )
%     cvx_end
    scales = x';

end
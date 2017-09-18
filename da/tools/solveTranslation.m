function x = solveTranslation(src, tgt)

    numConnections = size(src,1);
    numDims = size(src,2);
    
    % (E) Linear leasts squares to solve an overdetermined system
    A = repmat(sparse(eye(numDims,numDims)), [numConnections, 1]);
    b = tgt' - src';
    x = full(directSolutionSystemLinearEquations(A, b(:)))';
    
end
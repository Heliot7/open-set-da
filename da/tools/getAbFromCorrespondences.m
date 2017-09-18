function [A, b] = getAbFromCorrespondences(src, tgt)

    [numCorr, numDims] = size(src);
    
    A = spalloc(numDims*numDims,numCorr*numDims,numDims*(numCorr*numDims));
    b = zeros(numCorr*numDims,1);
    
    rowA = 1;
    for i = 1:numCorr
        for j = 1:numDims
            idxs = sub2ind([numDims,numDims],1:numDims,j*ones(1,numDims));
            A(idxs,rowA) = src(i,:);
            b(rowA) = tgt(i,j);
            rowA = rowA + 1;
        end
        fprintf('numCorr %d/%d\n',i,numCorr);
    end
    % A = A';

    % min_W 1/2 W'HW + qW (H = S'S and q = -S'T)
    % Solve the QP problem
%     H = A*A';
%     f = -A*b;
%     opts = optiset('display','iter');
%     Opt = opti('qp',H,f,'options',opts)
%     [x,fval,exitflag,info] = solve(Opt);
    
end


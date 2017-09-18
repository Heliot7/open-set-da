function W = estimateW(input, oldW, numDims, srcCentres, assignSrc, tgtCentres, assignTgt)

    W = oldW;    
    numInnerIter = input.numIterATI;
    for innerIt = 1:numInnerIter
        tic;
        fprintf('Starting transformation matrix W estimation\n');
        opt.algorithm = NLOPT_LD_MMA;
        opt.xtol_rel = 1e-5; %% 1e-2; , or 1
        opt.maxeval = input.numIterOpt;
        opt.verbose = 1; % 1;
        if(strcmpi(input.transformationDomain,'src'))
            opt.min_objective = (@(x) myfunc(x, numDims, srcCentres(assignSrc,:), tgtCentres(assignTgt,:)));
        else
            opt.min_objective = (@(x) myfunc(x, numDims, tgtCentres(assignTgt,:), srcCentres(assignSrc,:)));
        end
        xinit = W;
        xopt = nlopt_optimize(opt, xinit(:));
        W = reshape(xopt,numDims,numDims);
        toc;
        d = norm(srcCentres(assignSrc,:)*W - tgtCentres(assignTgt,:));
        fprintf('Full Matrix Optimisation (it %d/%d): energy %f\n', innerIt, numInnerIter, d);
    end

end

% Function handlers for objective functions, constraints and gradients 
function [val, gradient] = myfunc(x, numDims, srcCentres, tgtCentres)

    W = reshape(x,numDims,numDims);
    % Spectral norm
    LS = srcCentres*W - tgtCentres;
    val = 0.5*eigs(LS'*LS,1); % + regParam*norm(W);
    % Frobenius norm
% 	val = 0.5*norm(weights_mul.*(srcCentres*W - tgtCentres),'fro')^2;
%     val = 0.5*norm((srcCentres*W - tgtCentres),'fro')^2;
    if(nargout > 1)
        % 2*W*(X*X') - Y*X' - Y*X';
        srcCentres_T = srcCentres';
        gradient = srcCentres_T*srcCentres*W - srcCentres_T*tgtCentres;
        gradient = gradient(:);
    end
    
end

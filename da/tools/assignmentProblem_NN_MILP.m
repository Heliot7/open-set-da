function [assignsSrc, assignsTgt, fval] = assignmentProblem_NN_MILP(distST, distSS, NN, rho, lambda, fixed)

    fprintf('Preparing MILP...\n');

    numT = size(distST,1);
    numS = size(distST,2);
    numNN = size(NN, 2);

    % f'x
    f = [double(distST(:)); ones(numS*numT,1)];
    
    timeA = tic;
    % A*x < b
    % - A
    % -> A std part
    A = spalloc(numT+numT*numS,numT*numS+numT*numS,numT*numS*4+numT*numS*numNN*numS);
    line = ones(1,numS);
    for i = 1:numT
        idxs = sub2ind(size(distST), i*line, 1:numS);
        A(i,idxs) = 1;
    end
%     line = ones(1,numT);
%     for j = 1:numS
%         idxs = sub2ind(size(distST), 1:numT, j*line);
%         A(numT+j,idxs) = 1;
%     end
    % Constraints to ensure that at least 1 src cluster has a tgt sample
%     for j = 1:numS
%         idxs = sub2ind(size(distST), 1:numT, j*line);
%         A(numT+numS+j,idxs) = 1;
%     end
    % -> A linearisation
    constA = zeros(numT,numS); % a_ij loop OK
    for j = 1:numS 
        constA(:,j) = repmat(sum(size(NN,2).*distSS(j,:)), [numT 1]);
    end
    A(numT+1:numT+numT*numS,numT*numS+1:numT*numS+numT*numS) = -1*speye(numT*numS); % x_ij OK
    A(numT+1:numT+numT*numS,1:numT*numS) = spdiags(constA(:),0,numT*numS,numT*numS); % w_ij OK
    idx = 1;
    denseA = spalloc(numT*numS,numT*numS,numT*numS*numNN*numS); % x_i'j' OK
    for j = 1:numS
        for i = 1:numT
            for n = 1:numNN % NN(i,n) = t'
                if(NN(i,n) ~= 0)
                    idxs = sub2ind(size(distST),repmat(NN(i,n),[1 numS]),1:numS);
                    denseA(idxs, idx) = distSS(j,1:numS);
                end
            end
            idx = idx + 1;
        end
    end
    A(numT+1:numT+numT*numS,1:numT*numS) = A(numT+1:numT+numT*numS,1:numT*numS) + denseA'; % OK
    fprintf('Time spent in matrix A: %.0f ms\n', toc(timeA)*1000.0);
    
    if(lambda < 1.0)
        % - b
        b = [ones(numT,1); ... % tgt assigned to exactly 1 src
        constA(:)]; % ... % linearlisation constants (for NN > 0)
    else % all 1, std case
        % - b
        b = [ones(numT,1); constA(:)];
    end

    % - symbols
    e = [0*ones(numT,1); -1*ones(numT*numS,1)]; %-1 <=, 0 ==, 1 >=
    
    % variable types
    xtype = [repmat('B',1,numS*numT), repmat('C',1,numS*numT)];
    lb = zeros(numS*numT + numS*numT,1);
    ub = Inf*ones(numS*numT + numS*numT,1);
    % Add fixed label constraints
    for i = 1:length(fixed)
        lb(numT*(fixed(i)-1)+i) = 1;
        ub(numT*(fixed(i)-1)+i) = 1;
    end

    % Initial Guess with fixed values
    x0 = zeros(numS*numT + numS*numT, 1);
    for i = 1:length(fixed)
        x0(numT*(fixed(i)-1)+i) = 1;
    end
    
    % Run optimisation
    sopts = scipset('scipopts',{'limits/gap',0.0;'lp/threads',12});
    opts = optiset('display','iter','maxnodes',1000000,'maxtime',1500,'solver','scip','solverOpts',sopts);
    Opt = opti('f',f,'mix',A,b,e,'bounds',lb,ub,'xtype',xtype,'x0',x0,'options',opts);
    [x, fval] = solve(Opt);
    [assignsTgt, assignsSrc] = find(reshape(round(x(1:numT*numS)), [numT, numS]) == 1);
    % keyboard;
    
end

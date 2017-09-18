function [assignsSrc, assignsTgt, fval] = assignmentProblem_NN_BP(distST, distSS, NN, rho, lambda)

    numT = size(distST,1);
    numS = size(distST,2);
    numNN = size(NN,2);

    % f'x
    f = [double(distST(:)); zeros(numT*numS*numNN*numS,1)];
    
    % A*x < b
    % - A
    % -> A std part
    A = sparse(numT+numS+numT*numS*numNN*numS+1,numT*numS+numT*numS*numNN*numS);
    line = ones(1,numS);
    for i = 1:numT
       idxs = sub2ind(size(distST), i*line, 1:numS);
       A(i,idxs) = 1;
    end
    line = ones(1,numT);
    for j = 1:numS
        idxs = sub2ind(size(distST), 1:numT, j*line);
        A(numT+j,idxs) = 1;
    end
    % -> A linearisation
    idx = 1;
    for k = 1:numS
        for n = 1:numNN
            for j = 1:numS
                for i = 1:numT
                    % f_ijk
                    f(numT*numS+idx) = distSS(j,k);
                    % x_ij
                    A(numT+numS+idx,(j-1)*numT+i) = 1;
                    % x_N(i)k
                    A(numT+numS+idx,(k-1)*numT+NN(i,n)) = 1;
                    % w_ijk
                    A(numT+numS+idx,numT*numS+idx) = -2;
                    idx = idx + 1;
                end
            end
        end
    end
    A(end, numT*numS+1:end) = 1;
    numPlus = numT - floor(numT/(numS-1))*(numS-1); % -1 to remove emtpy node
    numMinus = numS-1 - numPlus;
    % - b
    b = [ones(numT,1); rho*ceil(numT/(numS-1))*ones(numPlus,1); rho*floor(numT/(numS-1))*ones(numMinus,1); lambda*numT; ...
        zeros(numT*numS*numNN*numS,1); numT*numNN];
    % - symbols
    e = [0*ones(numT,1); -1*ones(numS,1); ones(numT*numS*numNN*numS,1); 0]; % -1 <=, 0 ==, 1 >=
    
    % variable types
    xtype = repmat('B',1,numT*numS+numT*numS*numNN*numS);
    
    % Run optimisation
    opts = optiset('display','iter','maxnodes',100000,'maxtime',1000,'solver','scip');
    Opt = opti('f',f,'mix',A,b,e,'xtype',xtype,'options',opts);
    [x, fval] = solve(Opt);
    [assignsTgt, assignsSrc] = find(reshape(x(1:numT*numS), [numT, numS]));
    
end

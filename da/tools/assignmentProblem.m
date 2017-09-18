function [assignsSrc, assignsTgt, x] = assignmentProblem(distances, numCorr)

    numT = size(distances,1);
    numS = size(distances,2);
    
    f = double(distances(:));
    A = spalloc(numT+numS,numT*numS,numT*numS*2);
    line = ones(1,numS);
    for i = 1:numT
       idxs = sub2ind(size(distances), i*line, 1:numS);
       A(i,idxs) = 1;
    end
    line = ones(1,numT);
    for j = 1:numS
        idxs = sub2ind(size(distances), 1:numT, j*line);
        A(numT+j,idxs) = 1;
    end    
    numPlus = numT - floor(numT/numS)*numS;
    numMinus = numS - numPlus;
    b = [ones(numT,1); numCorr*ceil(numT/numS)*ones(numPlus,1); numCorr*floor(numT/numS)*ones(numMinus,1)];
    
    % - symbols
    e = [0*ones(numT,1); -1*ones(numS,1)]; %-1 <=, 0 ==, 1 >=
    
    xtype = repmat('B',1,length(f));
    
	opts = optiset('display','iter'); %,'solver','scip');
    Opt = opti('f',f,'mix',A,b,e,'xtype',xtype,'options',opts);
    % Opt = opti('f',f,'eq',A,b,'xtype',xtype,'options',opts);
    x = solve(Opt);
    [assignsTgt, assignsSrc] = find(reshape(x, [numT, numS]));
    keyboard;
    
end
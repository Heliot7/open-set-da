% DIJKSTRA Calculate Minimum Costs and Paths using Dijkstra's Algorithm
%
% Filename: dijkstra.m
%
% Description: Given vertices and edge connections for a graph (represented by
%       either adjacency matrix or edge list) and edge costs that are greater
%       than zero, this function computes the shortest path from one or more
%       starting nodes to one or more termination nodes.
%
% Author:
%       Joseph Kirk
%       jdkirk630@gmail.com
%
% Date: 02/27/15
%
% Release: 2.0
%
% Inputs:
%     [AorV] Either A or V where
%         A   is a NxN adjacency matrix, where A(I,J) is nonzero (=1)
%               if and only if an edge connects point I to point J
%               NOTE: Works for both symmetric and asymmetric A
%         V   is a Nx2 (or Nx3) matrix of x,y,(z) coordinates
%     [xyCorE] Either xy or C or E (or E3) where
%         xy  is a Nx2 (or Nx3) matrix of x,y,(z) coordinates (equivalent to V)
%               NOTE: only valid with A as the first input
%         C   is a NxN cost (perhaps distance) matrix, where C(I,J) contains
%               the value of the cost to move from point I to point J
%               NOTE: only valid with A as the first input
%         E   is a Px2 matrix containing a list of edge connections
%               NOTE: only valid with V as the first input
%         E3  is a Px3 matrix containing a list of edge connections in the
%               first two columns and edge weights in the third column
%               NOTE: only valid with V as the first input
%     [SID] (optional) 1xL vector of starting points
%         if unspecified, the algorithm will calculate the minimal path from
%         all N points to the finish point(s) (automatically sets SID = 1:N)
%     [FID] (optional) 1xM vector of finish points
%         if unspecified, the algorithm will calculate the minimal path from
%         the starting point(s) to all N points (automatically sets FID = 1:N)
%     [showWaitbar] (optional) a scalar logical that initializes a waitbar if nonzero
%
% Outputs:
%     [costs] is an LxM matrix of minimum cost values for the minimal paths
%     [paths] is an LxM cell array containing the shortest path arrays
%
% Note:
%     If the inputs are [A,xy] or [V,E], the cost is assumed to be (and is
%       calculated as) the point-to-point Euclidean distance
%     If the inputs are [A,C] or [V,E3], the cost is obtained from either
%       the C matrix or from the edge weights in the 3rd column of E3
%
% Usage:
%     [costs,paths] = dijkstra(A,xy)
%         -or-
%     [costs,paths] = dijkstra(A,C)
%         -or-
%     [costs,paths] = dijkstra(V,E)
%         -or-
%     [costs,paths] = dijkstra(V,E3)
%         -or-
%     [costs,paths] = dijkstra( ... ,SID,FID)
%         -or-
%     [costs,paths] = dijkstra( ... ,SID,FID,true)
%
% Example:
%     % Calculate the (all pairs) shortest distances and paths using [A,xy] inputs
%     n = 7; A = zeros(n); xy = 10*rand(n,2)
%     tri = delaunay(xy(:,1),xy(:,2));
%     I = tri(:); J = tri(:,[2 3 1]); J = J(:);
%     IJ = I + n*(J-1); A(IJ) = 1
%     [costs,paths] = dijkstra(A,xy)
%
% Example:
%     % Calculate the (all pairs) shortest distances and paths using [A,C] inputs
%     n = 7; A = zeros(n); xy = 10*rand(n,2)
%     tri = delaunay(xy(:,1),xy(:,2));
%     I = tri(:); J = tri(:,[2 3 1]); J = J(:);
%     IJ = I + n*(J-1); A(IJ) = 1
%     a = (1:n); b = a(ones(n,1),:);
%     C = round(reshape(sqrt(sum((xy(b,:) - xy(b',:)).^2,2)),n,n))
%     [costs,paths] = dijkstra(A,C)
%
% Example:
%     % Calculate the (all pairs) shortest distances and paths using [V,E] inputs
%     n = 7; V = 10*rand(n,2)
%     I = delaunay(V(:,1),V(:,2));
%     J = I(:,[2 3 1]); E = [I(:) J(:)]
%     [costs,paths] = dijkstra(V,E)
%
% Example:
%     % Calculate the (all pairs) shortest distances and paths using [V,E3] inputs
%     n = 7; V = 10*rand(n,2)
%     I = delaunay(V(:,1),V(:,2));
%     J = I(:,[2 3 1]);
%     D = sqrt(sum((V(I(:),:) - V(J(:),:)).^2,2));
%     E3 = [I(:) J(:) D]
%     [costs,paths] = dijkstra(V,E3)
%
% Example:
%     % Calculate the shortest distances and paths from the 3rd point to all the rest
%     n = 7; V = 10*rand(n,2)
%     I = delaunay(V(:,1),V(:,2));
%     J = I(:,[2 3 1]); E = [I(:) J(:)]
%     [costs,paths] = dijkstra(V,E,3)
%
% Example:
%     % Calculate the shortest distances and paths from all points to the 2nd
%     n = 7; A = zeros(n); xy = 10*rand(n,2)
%     tri = delaunay(xy(:,1),xy(:,2));
%     I = tri(:); J = tri(:,[2 3 1]); J = J(:);
%     IJ = I + n*(J-1); A(IJ) = 1
%     [costs,paths] = dijkstra(A,xy,1:n,2)
%
% Example:
%     % Calculate the shortest distance and path from points [1 3 4] to [2 3 5 7]
%     n = 7; V = 10*rand(n,2)
%     I = delaunay(V(:,1),V(:,2));
%     J = I(:,[2 3 1]); E = [I(:) J(:)]
%     [costs,paths] = dijkstra(V,E,[1 3 4],[2 3 5 7])
%
% Example:
%     % Calculate the shortest distance and path between two points
%     n = 1000; A = zeros(n); xy = 10*rand(n,2);
%     tri = delaunay(xy(:,1),xy(:,2));
%     I = tri(:); J = tri(:,[2 3 1]); J = J(:);
%     D = sqrt(sum((xy(I,:)-xy(J,:)).^2,2));
%     I(D > 0.75,:) = []; J(D > 0.75,:) = [];
%     IJ = I + n*(J-1); A(IJ) = 1;
%     [cost,path] = dijkstra(A,xy,1,n)
%     gplot(A,xy,'k.:'); hold on;
%     plot(xy(path,1),xy(path,2),'ro-','LineWidth',2); hold off
%     title(sprintf('Distance from %d to %d = %1.3f',1,n,cost))
%
% Web Resources:
%   <a href="http://en.wikipedia.org/wiki/Dijkstra%27s_algorithm">Dijkstra's Algorithm</a>
%   <a href="http://en.wikipedia.org/wiki/Graph_%28mathematics%29">Graphs</a>
%   <a href="http://en.wikipedia.org/wiki/Adjacency_matrix">Adjacency Matrix</a>
%
% See also: gplot, gplotd, gplotdc, distmat, ve2axy, axy2ve
%
function [costs,paths] = dijkstra(AorV,xyCorE,SID,FID,showWaitbar)
    
    narginchk(2,5);
    
    % Process inputs
    [n,nc] = size(AorV);
    [m,mc] = size(xyCorE);
    if (nargin < 3)
        SID = (1:n);
    elseif isempty(SID)
        SID = (1:n);
    end
    L = length(SID);
    if (nargin < 4)
        FID = (1:n);
    elseif isempty(FID)
        FID = (1:n);
    end
    M = length(FID);
    if (nargin < 5)
        showWaitbar = (n > 1000 && max(L,M) > 1);
    end
    
    
    % Error check inputs
    if (max(SID) > n || min(SID) < 1)
        eval(['help ' mfilename]);
        error('Invalid [SID] input. See help notes above.');
    end
    if (max(FID) > n || min(FID) < 1)
        eval(['help ' mfilename]);
        error('Invalid [FID] input. See help notes above.');
    end
    [E,cost] = process_inputs(AorV,xyCorE);
    
    
    % Reverse the algorithm if it will be more efficient
    isReverse = false;
    if L > M
        E = E(:,[2 1]);
        cost = cost';
        tmp = SID;
        SID = FID;
        FID = tmp;
        isReverse = true;
    end
    
    
    % Initialize output variables
    L = length(SID);
    M = length(FID);
    costs = zeros(L,M);
    paths = num2cell(NaN(L,M));
    
    
    % Create a waitbar if desired
    if showWaitbar
        hWait = waitbar(0,'Calculating Minimum Paths ... ');
    end
    
    
    % Find the minimum costs and paths using Dijkstra's Algorithm
    for k = 1:L
        
        % Initializations
        iTable = NaN(n,1);
        minCost = Inf(n,1);
        isSettled = false(n,1);
        path = num2cell(NaN(n,1));
        I = SID(k);
        minCost(I) = 0;
        iTable(I) = 0;
        isSettled(I) = true;
        path(I) = {I};
        
        % Execute Dijkstra's Algorithm for this vertex
        while any(~isSettled(FID))
            
            % Update the table
            jTable = iTable;
            iTable(I) = NaN;
            nodeIndex = find(E(:,1) == I);
            
            % Calculate the costs to the neighbor nodes and record paths
            for kk = 1:length(nodeIndex)
                J = E(nodeIndex(kk),2);
                if ~isSettled(J)
                    c = cost(I,J);
                    empty = isnan(jTable(J));
                    if empty || (jTable(J) > (jTable(I) + c))
                        iTable(J) = jTable(I) + c;
                        if isReverse
                            path{J} = [J path{I}];
                        else
                            path{J} = [path{I} J];
                        end
                    else
                        iTable(J) = jTable(J);
                    end
                end
            end
            
            % Find values in the table
            K = find(~isnan(iTable));
            if isempty(K)
                break
            else
                % Settle the minimum value in the table
                [~,N] = min(iTable(K));
                I = K(N);
                minCost(I) = iTable(I);
                isSettled(I) = true;
            end
        end
        
        % Store costs and paths
        costs(k,:) = minCost(FID);
        paths(k,:) = path(FID);
        if showWaitbar && ~mod(k,ceil(L/100))
            waitbar(k/L,hWait);
        end
    end
    if showWaitbar
        delete(hWait);
    end
    
    
    % Reformat outputs if algorithm was reversed
    if isReverse
        costs = costs';
        paths = paths';
    end
    
    
    % Pass the path as an array if only one source/destination were given
    if L == 1 && M == 1
        paths = paths{1};
    end
    
    
    % -------------------------------------------------------------------
    function [E,C] = process_inputs(AorV,xyCorE)
        if (n == nc)
            if (m == n)
                A = AorV;
                A = A - diag(diag(A));
                if (m == mc)
                    % Inputs = (A,cost)
                    C = xyCorE;
                    E = a2e(A);
                else
                    % Inputs = (A,xy)
                    xy = xyCorE;
                    E = a2e(A);
                    D = ve2d(xy,E);
                    C = sparse(E(:,1),E(:,2),D,n,n);
                end
            else
                eval(['help ' mfilename]);
                error('Invalid [A,xy] or [A,C] inputs. See help notes above.');
            end
        else
            if (mc == 2)
                % Inputs = (V,E)
                V = AorV;
                E = xyCorE;
                D = ve2d(V,E);
                C = sparse(E(:,1),E(:,2),D,n,n);
            elseif (mc == 3)
                % Inputs = (V,E3)
                E3 = xyCorE;
                E = E3(:,1:2);
                C = sparse(E(:,1),E(:,2),E3(:,3),n,n);
            else
                eval(['help ' mfilename]);
                error('Invalid [V,E] or [V,E3] inputs. See help notes above.');
            end
        end
    end
    
end

% Convert adjacency matrix to edge list
function E = a2e(A)
    [I,J] = find(A);
    E = [I J];
end

% Compute Euclidean distance for edges
function D = ve2d(V,E)
    VI = V(E(:,1),:);
    VJ = V(E(:,2),:);
    D = sqrt(sum((VI - VJ).^2,2));
end


function [w_st, ST, X_st] = kruskal(X, w)
% function [w_st, ST, X_st] = kruskal(X, w)
%
% This function finds the minimum spanning tree of the graph where each
% edge has a specified weight using the Kruskal's algorithm.
% 
% Assumptions
% -----------
%     N:  1x1  scalar      -  Number of nodes (vertices) of the graph
%    Ne:  1x1  scalar      -  Number of edges of the graph
%   Nst:  1x1  scalar      -  Number of edges of the minimum spanning tree
% 
% We further assume that the graph is labeled consecutively. That is, if
% there are N nodes, then nodes will be labeled from 1 to N.
%
% INPUT
% 
%     X:  NxN logical      -  Adjacency matrix
%             matrix          If X(i,j)=1, this means there is directed edge
%                             starting from node i and ending in node j.
%                             Each element takes values 0 or 1.
%                             If X symmetric, graph is undirected.
% 
%  or     Nex2 double      -  Neighbors' matrix
%              matrix         Each row represents an edge.
%                             Column 1 indicates the source node, while
%                             column 2 the target node.
% 
%     w:  NxN double       -  Weight matrix in adjacency form
%             matrix          If X symmetric (undirected graph), w has to
%                             be symmetric.
% 
%  or     Nex1 double      -  Weight matrix in neighbors' form
%              matrix         Each element represents the weight of that
%                             edge.
% 
% 
% OUTPUT
% 
%  w_st:    1x1 scalar     -  Total weight of minimum spanning tree
%    ST:  Nstx2 double     -  Neighbors' matrix of minimum spanning tree
%               matrix
%  X_st:  NstxNst logical  -  Adjacency matrix of minimum spanning tree
%                 matrix      If X_st symmetric, tree is undirected.
% 
% EXAMPLES
%
% Undirected graph
% ----------------
% Assume the undirected graph with adjacency matrix X and weights w:
%
%         1   
%       /   \
%      2     3
%     / \
%    4 - 5
% 
% X = [0 1 1 0 0;
%      1 0 0 1 1;
%      1 0 0 0 0;
%      0 1 0 0 1;
%      0 1 0 1 0];
%  
% w = [0 1 2 0 0;
%      1 0 0 2 1;
%      2 0 0 0 0;
%      0 2 0 0 3;
%      0 1 0 3 0];
% 
% [w_st, ST, X_st] = kruskal(X, w);
% The above function gives us the minimum spanning tree.
% 
% 
% Directed graph
% ----------------
% Assume the directed graph with adjacency matrix X and weights w:
%
%           1
%        / ^ \
%       / /   \
%      v       v
%       2 ---> 3
% 
% X = [0 1 1
%      1 0 1
%      0 0 0];
%  
% w = [0 1 4;
%      2 0 1;
%      0 0 0];
% 
% [w_st, ST, X_st] = kruskal(X, w);
% The above function gives us the minimum directed spanning tree.
% 
% 
% Author: Georgios Papachristoudis
% Copyright 2013 Georgios Papachristoudis
% Date: 2013/05/26 12:25:18

    isUndirGraph = 1;
    
    % Convert logical adjacent matrix to neighbors' matrix    
    if size(X,1)==size(X,2) && sum(X(:)==0)+sum(X(:)==1)==numel(X)        
        if any(any(X-X'))
            isUndirGraph = 0;
        end
        ne = cnvrtX2ne(X,isUndirGraph);
    else
        if size(unique(sort(X,2),'rows'),1)~=size(X,1)
            isUndirGraph = 0;
        end
        ne = X;
    end
    
    % Convert weight matrix from adjacent to neighbors' form
    if numel(w)~=length(w)
        if isUndirGraph && any(any(w-w'))
            error('If it is an undirected graph, weight matrix has to be symmetric.');
        end
        w = cnvrtw2ne(w,ne);
    end
    
    N    = max(ne(:));   % number of vertices
    Ne   = size(ne,1);   % number of edges    
    lidx = zeros(Ne,1);  % logical edge index; 1 for the edges that will be
                         % in the minimum spanning tree                         
    % Sort edges w.r.t. weight
    [w,idx] = sort(w);
    ne      = ne(idx,:);
    
    % Initialize: assign each node to itself
    [repr, rnk] = makeset(N);
    
    % Run Kruskal's algorithm
    for k = 1:Ne
        i = ne(k,1);
        j = ne(k,2);
        if fnd(i,repr) ~= fnd(j,repr)
            lidx(k) = 1;
            [repr, rnk] = union(i, j, repr, rnk);
        end
    end
    
    % Form the minimum spanning tree
    treeidx = find(lidx);
    ST      = ne(treeidx,:);
    
    % Generate adjacency matrix of the minimum spanning tree
    X_st = zeros(N);
    for k = 1:size(ST,1)
        X_st(ST(k,1),ST(k,2)) = 1;
        if isUndirGraph,  X_st(ST(k,2),ST(k,1)) = 1;  end
    end
    
    % Evaluate the total weight of the minimum spanning tree
    w_st = sum(w(treeidx));
end

function ne = cnvrtX2ne(X, isUndirGraph)
    if isUndirGraph
        ne = zeros(sum(sum(X.*triu(ones(size(X))))),2);
    else
        ne = zeros(sum(X(:)),2);
    end
    cnt = 1;
    for i = 1:size(X,1)
        v       = find(X(i,:));        
        if isUndirGraph
            v(v<=i) = [];
        end
        u       = repmat(i, size(v));
        edges   = [u; v]';
        ne(cnt:cnt+size(edges,1)-1,:) = edges;
        cnt = cnt + size(edges,1);
    end
end

function w = cnvrtw2ne(w,ne)
    tmp = zeros(size(ne,1),1);
    cnt = 1;
    for k = 1:size(ne,1)
        tmp(cnt) = w(ne(k,1),ne(k,2));
        cnt = cnt + 1;
    end
    w = tmp;
end

function [repr, rnk] = makeset(N)
    repr = (1:N);
    rnk  = zeros(1,N);
end

function o = fnd(i,repr)
    while i ~= repr(i)
        i = repr(i);
    end
    o = i;
end

function [repr, rnk] = union(i, j, repr, rnk)
    r_i = fnd(i,repr);
    r_j = fnd(j,repr);
    if rnk(r_i) > rnk(r_j)
        repr(r_j) = r_i;
    else
        repr(r_i) = r_j;
        if rnk(r_i) == rnk(r_j)
            rnk(r_j) = rnk(r_j) + 1;
        end
    end
end
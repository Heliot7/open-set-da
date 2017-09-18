function [coeff, scores] = mPCA(data)

    % Normalised & 0-mean
    % data = zscore(data);
    % data = data - repmat(mean(data),[size(data,1) 1]);
    
    covM = cov(data);
    [V, D] = eig(covM);
    D = diag(D);
    [~, idx] = sort(D, 'descend');
    coeff = V(:,idx);
    scores = data * coeff;
    
end


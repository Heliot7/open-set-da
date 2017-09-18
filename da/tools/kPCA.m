function coeff = kPCA(data, k)

    % Assume mean is already substracted
    [~,~, coeff] = randomizedSVD(data, k);
    % [~,~, coeff] = fsvd(data, k, 2, true);
    
end


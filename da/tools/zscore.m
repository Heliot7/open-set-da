function [x, mu, sigma] = zscore(x, mu, sigma)

    if(nargin <= 1)
        sigma = max(std(x),eps);
        mu = mean(x);
    end
    x = bsxfun(@minus,x,mu);
    x = bsxfun(@rdivide,x,sigma);
    
end
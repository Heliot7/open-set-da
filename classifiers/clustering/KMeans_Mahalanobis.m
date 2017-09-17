function [means, covs, labels] = KMeans_Mahalanobis(x, numClusters, numTries)

    x = rand(1000, 256, 'single'); % Remove
    numClusters = 31; % Remove
    
    numSamples = size(x,1);
    numDims = size(x,2);
    
    x = x .* repmat(rand(1, numDims)*10, [numSamples 1]) - 5; % Remove

    t = tic;
    oldEnergy = Inf;
    dist = zeros(numSamples, numClusters, 'single');
%     dist = zeros(numSamples, numClusters);
    dist2 = zeros(numSamples, numClusters);
    for it = 1:numTries
        
        % Select random initial distributions
        means = x(randi(numSamples, [numClusters 1]), :);
        covs = repmat(reshape(eye(numDims, 'single'), [1 numDims numDims]),[numClusters 1 1]);
%         covs = repmat(reshape(eye(numDims), [1 numDims numDims]),[numClusters 1 1]);

        numChanges = Inf;
        oldTags = zeros(numSamples, 1);
        while(numChanges > 0.05*numSamples)

            % Assign distribution for each sample    
            for c = 1:numClusters
%                 mu = means(c,:);
%                 ep = squeeze(covs(c,:,:));
%                 for s = 1:numSamples
%                     dist(s,c) = abs(sum((x(s,:) - mu).*(x(s,:) - mu) / ep));
%                 end
                subMean = num2cell(x - repmat(means(c,:),[numSamples 1]), 2);
%                 invC = inv(squeeze(covs(c,:,:)));
%                 dist(:,c) = abs(cellfun(@(p) p*invC*p', subMean));
                C = squeeze(covs(c,:,:));
                if(rank(C) < C)
                    % singular covariance matrix
                    keyboard;
                end
                dist(:,c) = abs(cellfun(@(p) sum(p.*p/C), subMean));
            end
            [allEnergies, tags] = min(dist,[],2);
            energy = sum(allEnergies);

            numChanges = sum(tags ~= oldTags);
            oldTags = tags;

            % Recompute distributions
            for c = 1:numClusters
                samples = x(tags == c, :);
                if(size(samples,1) > 1)
                    means(c,:) = median(samples);
                    covs(c,:,:) = cov(samples);
                elseif(size(samples,1) == 1)
                    % Issues!
                    means(c,:) = median(samples);
                    covs(c,:,:) = cov(samples);
                    keyboard;                    
                else
                    % Issues!
                    keyboard;
                end
            end

        end
    
        if(energy < oldEnergy)
            labels = tags;
            oldEnergy = energy;
            % fprintf('[it %d] new energy: %.2f\n', it, energy);
        end
        
    end
    fprintf('Total KMeans_Mahalanobis time: %.2fsec\n', toc(t));

    % Plots results
    coeff = mPCA(x);
    xPCA = x*coeff(:,1:2); 
    
    figure;
    scatter(xPCA(:,1), xPCA(:,2), 25, labels);
    
end


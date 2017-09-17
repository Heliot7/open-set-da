function [data, features, testData, testFeatures] = getSentiment(input, dataset_info, phase)

    testFeatures = [];
    testData = [];
    
    % NOTE: Change everytime we run new experiments to get new results
    rng(input.seedRand);
    % Get random splits
    load([input.PATH_DATA dataset_info.path 'gong\' dataset_info.(phase) '_400_idx']);
    if(strcmpi(phase,'source'))
        splits = train_idx(:,input.seedRand);
    else % target
        splits = test_idx(:,input.seedRand);
    end

    % If features previously computed
    load([input.PATH_DATA dataset_info.path 'gong\' dataset_info.(phase) '_400']);
    fts = fts(splits,:);
    if(input.isZScore)
        fts = fts ./ repmat(sum(fts,2),1,size(fts,2));
        fts = zscore(fts);
    end
    features = fts;    
    % Get nunImgs from all of them
    posIds = ismember(splits,find((labels == 1)));
    f_pos = features(posIds,:);
    negIds = ismember(splits,find((labels == 0)));
    f_neg = features(negIds,:);
    features = [f_pos; f_neg];
    data.annotations.classes = [repmat({'pos'}, [sum(posIds) 1]); repmat({'neg'}, [sum(negIds) 1])];

end

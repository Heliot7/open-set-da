function [data, features, testData, testFeatures] = getSaenkoPrecomputed(input, dataset_info, phase, numClasses)

    features = []; testFeatures = [];
    data.annotations.classes = []; testData.annotations.classes = [];
    path = [input.PATH_DATA dataset_info.path];
    dataset = dataset_info.(phase);
    if(strcmpi(phase,'source'))
        if(input.daAllSrc) % if(input.seedRand == -1)
            numImgs = 99999;
        else
            numImgs = 20;
            if(strcmpi(dataset,'WEBCAM') || strcmpi(dataset,'DSLR'))
                numImgs = 8;
            end
        end
    elseif(strcmpi(phase,'target'))
        if(input.isClassSupervised)
            numImgs = 3;
        else % unsupervised
            numImgs = 99999;
        end
    end

    % NOTE: Change everytime we run new experiments to get new results
    if(input.seedRand > 0)
        rng(input.seedRand);
    end

    if(strcmpi(input.typeDescriptor,'BoW'))
        load([path lower(dataset) '_SURF_L' num2str(numClasses)]);
    elseif(~isempty(strfind(input.typeDescriptor,'CNN-fc')))
        if(~isempty(strfind(input.typeDescriptor,'FT')))
            load([path 'office_fc7_FT\' input.cnnName '-' input.sourceDataset.source '_' lower(dataset) '_CNN' input.typeDescriptor(4:end) '_L' num2str(numClasses)]);
        else
            load([path lower(dataset) '_CNN' input.typeDescriptor(4:end) '_L' num2str(numClasses)]);
        end
    else
        error('[[Catched ERROR: No precomputed features]]');
    end
    if(input.isZScore)
        fts = fts./ repmat(sum(fts,2),1,size(fts,2));
        fts = zscore(fts);
    end
    
    if(~input.isOpenset)
        listLabels = 1:max(labels);
    else
        if(strcmpi(phase,'source'))
            listLabels = 1:20;
        else
            listLabels = [1:10,21:max(labels)];
        end
    end
    
    for i = listLabels
        classSamples = find(labels == i);
        permSamples = randperm(length(classSamples));
        % Train portion
        trainSamples = classSamples(permSamples(1:min(length(classSamples), numImgs)));
        features = [features; fts(trainSamples,:)];
        % To comment (closed set -> open set)
        if(input.isOpenset || input.isWSVM)
            if(i <= 10) 
                data.annotations.classes = [data.annotations.classes; dataset_info.classes(labels(trainSamples))'];
            else
                data.annotations.classes = [data.annotations.classes;  repmat({'zz_unknown'}, [size(trainSamples,1) 1])];
            end
        else
            data.annotations.classes = [data.annotations.classes; dataset_info.classes(labels(trainSamples))'];
        end
        % Test portion
        if(numImgs < length(classSamples) && strcmpi(phase,'target'))
            testSamples = classSamples(permSamples(numImgs+1:end));
            testFeatures = [testFeatures; fts(testSamples,:)];
            if(input.isOpenset || input.isWSVM)
                if(i <= 10) 
                    testData.annotations.classes = [testData.annotations.classes; dataset_info.classes(labels(testSamples))'];
                else
                    testData.annotations.classes = [testData.annotations.classes;  repmat({'zz_unknown'}, [size(testSamples,1) 1])];
                end
            else
                testData.annotations.classes = [testData.annotations.classes; dataset_info.classes(labels(testSamples))'];
            end
        end
    end
    
    if(isempty(testData.annotations.classes))
        testData = [];
    end
    
end

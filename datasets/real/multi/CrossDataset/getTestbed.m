function [data, features, testData, testFeatures] = getTestbed(input, dataset_info, phase)

    path = [input.PATH_DATA dataset_info.path];
    dataset = dataset_info.(phase);
    if(strcmpi(phase,'source'))
        if(input.daAllSrc)
            numImgs = 99999;
        else
            numImgs = 50; % 10, 20, 30, 40, 50
        end
    elseif(strcmpi(phase,'target'))
        % if(~input.daAllTgt)
            if(strcmpi(dataset,'CALTECH256'))
                numImgs = 30;
            else
                numImgs = 20;
            end
        % else
            % numImgs = 9999;
        % end
        if(input.isClassSupervised)
            numImgs = 3;
        end
    end

    % NOTE: Change everytime we run new experiments to get new results
    rng(input.seedRand); 
    if(strcmpi(input.typeDescriptor,'CNN-fc7'))
        load([path 'dense_setup_decaf7\dense_' lower(dataset) '_decaf7']);
    else
        load([path 'dense_setup_siftBOW\dense_' lower(dataset) '_sift']);
    end
    if(input.isZScore)
        fts = fts ./ repmat(sum(fts,2),1,size(fts,2)); 
        fts = zscore(fts);
    end    
    
    numClasses = max(label);
    if(~input.isRebuttal)
        listClasses = 1:numClasses;
    else
        if(~input.isOpenset)
            listClasses = 1:10;
            if(strcmpi(phase,'source') && input.isWSVM)
                listClasses = 1:10;
            elseif(strcmpi(phase,'target') && input.isWSVM)
                listClasses = [1:10,26:numClasses];
            end
        else
            if(strcmpi(phase,'source'))
                listClasses = 1:25;
            else
                listClasses = [1:10,26:numClasses];
            end
        end
    end
    
    features = []; testFeatures = [];
    data.annotations.classes = [];
    testData.annotations.classes = [];
    
    for i = listClasses
        classSamples = find(label == i);
        permSamples = randperm(length(classSamples));
        % Train portion
        trainSamples = classSamples(permSamples(1:min(length(classSamples), numImgs)));
        features = [features; fts(trainSamples,:)];
        if(input.isRebuttal && (input.isOpenset || input.isWSVM))
            if(i <= 10) 
                data.annotations.classes = [data.annotations.classes; dataset_info.classes(label(trainSamples))'];
            else
                data.annotations.classes = [data.annotations.classes;  repmat({'background'}, [size(trainSamples,1) 1])];
            end
        else
            data.annotations.classes = [data.annotations.classes; dataset_info.classes(label(trainSamples))'];
        end
        % data.annotations.classes = [data.annotations.classes; dataset_info.classes(label(trainSamples))'];
        % Test portion
        if(numImgs < length(classSamples) && (strcmpi(phase,'target') && input.isClassSupervised))
            testSamples = classSamples(permSamples(numImgs+1:end));
            testFeatures = [testFeatures; fts(testSamples,:)];
            if(input.isRebuttal && (input.isOpenset || input.isWSVM))
                if(i <= 10) 
                    testData.annotations.classes = [testData.annotations.classes; dataset_info.classes(label(testSamples))'];
                else
                    testData.annotations.classes = [testData.annotations.classes;  repmat({'background'}, [size(testSamples,1) 1])];
                end
            else
                testData.annotations.classes = [testData.annotations.classes; dataset_info.classes(label(testSamples))'];
            end
            % testData.annotations.classes = [testData.annotations.classes; dataset_info.classes(label(testSamples))'];
        end
    end    

    if(isempty(testData.annotations.classes))
        testData = [];
    end
    
end


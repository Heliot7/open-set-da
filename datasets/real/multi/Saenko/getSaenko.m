function [data, testData] = getSaenko(input, dataset_info, phase, numClasses)

    dataset = dataset_info.(phase);
    if(strcmpi(phase,'source'))
        if(input.daAllSrc) % input.seedRand == -1)
            numImgs = 99999; % (fully-transductive protocol)
        else
            numImgs = 20;
            if(strcmpi(dataset,'WEBCAM') || strcmpi(dataset,'DSLR'))
                numImgs = 8;
            end
        end
    elseif(strcmpi(phase,'target'))
        if(input.isClassSupervised)
            numImgs = 3; % (should be 3 in protocol)
        else % unsupervised
            numImgs = 99999;
        end
    end
    
    % NOTE: Change everytime we run new experiments to get new results
    % Protocol: 5 random split tests
    if(input.seedRand > 0)
        rng(input.seedRand);
    end

    if(~input.isOpenset)
        listClasses = 1:numClasses;
    else
        if(strcmpi(phase,'source'))
            listClasses = 1:20;
        else
            listClasses = [1:10,21:numClasses];
        end
    end
    
    path = [input.PATH_DATA dataset_info.path lower(dataset) '\'];
    folder = dir(path);
    allClassFolders = {folder.name};
    [~, ~, exts] = cellfun(@fileparts, allClassFolders, 'UniformOutput', false);
    idxImgPaths = cellfun(@(x) sum(x), cellfun(@(x) strcmp(x, {''}), exts, 'UniformOutput', false)) == 1;
    allClassFolders = allClassFolders(idxImgPaths)';
    
    data.imgPaths = []; data.annotations.classes = [];
    data.annotations.imgId = [];
    % data.annotations.BB = [];
    testData.imgPaths = []; testData.annotations.classes = [];
    testData.annotations.imgId = [];
    % testData.annotations.BB = [];
    for i = listClasses
        
        % Select image paths
        folder = dir([path allClassFolders{i}]);
        classFolders = {folder.name};
        [~, ~, exts] = cellfun(@fileparts, classFolders, 'UniformOutput', false);
        idxImgPaths = cellfun(@(x) sum(x), cellfun(@(x) strcmp(x, {'.jpg'}), exts, 'UniformOutput', false)) == 1;
        classFolders = classFolders(idxImgPaths)';
        numSamples = min(length(classFolders), numImgs);
        permSamples = randperm(length(classFolders));
        data.annotations.imgId = [data.annotations.imgId; length(data.imgPaths)+(1:numSamples)'];
        data.imgPaths = [data.imgPaths; strcat(repmat({[path allClassFolders{i} '\']}, ...
            [numSamples 1]), classFolders(permSamples(1:numSamples)))];
        % To comment (closed set -> open set)
        if(input.isRebuttal && (input.isOpenset || input.isWSVM))
            if(i <= 10) 
                data.annotations.classes = [data.annotations.classes; dataset_info.classes(i*ones(numSamples,1))'];
            else
                data.annotations.classes = [data.annotations.classes;  repmat({'background'}, [numSamples 1])];
            end
        else
            data.annotations.classes = [data.annotations.classes; dataset_info.classes(i*ones(numSamples,1))'];
        end
        % data.annotations.BB = [data.annotations.BB; zeros(numSamples,0)];
        
        % Test portion
        if(numSamples < length(classFolders) && strcmpi(phase, 'target'))
            testSamples = classFolders(permSamples(numImgs+1:end));
            numSamples = length(testSamples);
            % testData.annotations.imgId = [testData.annotations.imgId; length(data.imgPaths)+(1:numSamples)'];
            testData.imgPaths = [testData.imgPaths; strcat(repmat({[path allClassFolders{i} '\']}, ...
            [numSamples 1]), testSamples)];
            if(input.isRebuttal && (input.isOpenset || input.isWSVM))
                if(i <= 10) 
                    testData.annotations.classes = [testData.annotations.classes; dataset_info.classes(i*ones(numSamples,1))'];
                else
                    testData.annotations.classes = [testData.annotations.classes;  repmat({'background'}, [numSamples 1])];
                end
            else
                testData.annotations.classes = [testData.annotations.classes; dataset_info.classes(i*ones(numSamples,1))'];
            end
            % testData.annotations.BB = [testData.annotations.BB; zeros(numSamples,4)];
        end
    end
    testData.annotations.imgId = (1:length(testData.imgPaths))';
    
    if(isempty(testData.annotations.classes))
        testData = [];
    end
    
    % Save txt files with random splits
    fileID = -1;
    strOS = '';
    if(input.isRebuttal)
        if(input.isOpenset)
            strOS = 'OS_';
        elseif(input.isWSVM)
            strOS = 'CSOS_';
        else
            strOS = 'OSno_';
        end
    end
    if(strcmpi(phase,'source'))
        save_file = sprintf('D:/Core/Caffe-DA/data/office/%s%s_source_rnd%d.txt', strOS, lower(dataset), input.seedRand);
        fileID = fopen(save_file,'w');
    elseif(strcmpi(phase,'target'))
        strSup = 'uns';
        if(input.isClassSupervised)
            strSup = 'sup';
        end
        save_file = sprintf('D:/Core/Caffe-DA/data/office/%s%s_train_%s_rnd%d.txt', strOS, lower(dataset), strSup, input.seedRand); 
        fileID = fopen(save_file,'w');
        if(input.isClassSupervised || (~input.isClassSupervised && input.seedRand == -1))
            save_file_test = sprintf('D:/Core/Caffe-DA/data/office/%s%s_test_%s_rnd%d.txt', strOS, lower(dataset), strSup, input.seedRand); 
            fileID_test = fopen(save_file_test,'w');
        end        
    end
    
    if(fileID > 0)
        formatSpec = '%s %d\n';
        for idx = 1:length(data.imgPaths)
            id = find(ismember(dataset_info.classes, data.annotations.classes{idx}));
            if(isempty(id)) % open set background)
                id = 11;
            end
            fprintf(fileID, formatSpec, strrep(data.imgPaths{idx},'\', '/'), id-1);
        end
        fclose(fileID);
        if(strcmpi(phase,'target') && (input.isClassSupervised ||(~input.isClassSupervised && input.seedRand == -1)))
            if(input.isClassSupervised)
                for idx = 1:length(testData.imgPaths)
                    id = find(ismember(dataset_info.classes, testData.annotations.classes{idx}));
                    if(isempty(id)) % open set background)
                        id = 11;
                    end
                    fprintf(fileID_test, formatSpec, strrep(testData.imgPaths{idx},'\', '/'), id-1);
                end
            elseif(~input.isClassSupervised && input.seedRand == -1)
                for idx = 1:length(data.imgPaths)
                    id = find(ismember(dataset_info.classes, data.annotations.classes{idx}));
                    if(isempty(id)) % open set background)
                        id = 11;
                    end
                    fprintf(fileID_test, formatSpec, strrep(data.imgPaths{idx},'\', '/'), id-1);
                end
            end
            fclose(fileID_test);
        end
    end

end

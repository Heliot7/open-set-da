function [data, features, testData, testFeatures] = getCrossDataset_raw(input, dataset_info, phase)

    dataset = lower(dataset_info.(phase));
    if(strcmpi(phase,'source'))
        if(input.daAllSrc)
            numImgs = 99999;
        else
            if(strcmpi(dataset,'eth80') || strcmpi(dataset,'office') || strcmpi(dataset,'msrcorid'))
                numImgs = 9999;
            elseif(strcmpi(dataset,'pascal07') || strcmpi(dataset,'caltech101'))
                numImgs = 9999;
            else
                numImgs = 9999;
            end
            
        end
    elseif(strcmpi(phase,'target'))
        % if(input.daAllTgt)
        %     numImgs = 100;
        % else
        if(strcmpi(dataset,'eth80') || strcmpi(dataset,'office') || strcmpi(dataset,'msrcorid'))
            numImgs = 9999;
        elseif(strcmpi(dataset,'pascal07') || strcmpi(dataset,'caltech101'))
            numImgs = 9999;
        else
            numImgs = 9999;
        end
        % end
        if(input.isClassSupervised)
            numImgs = 3;
        end
    end

    % NOTE: Change everytime we run new experiments to get new results
    rng(input.seedRand); 

    % New features for sparse classification not available yet
    % parseLabels(input, dataset, path);
    
    file = fopen([input.PATH_DATA input.([phase 'Dataset']).path 'sparse_' dataset '.txt']);
    fileData = textscan(file, '%s %s');
    fclose(file);
    namePathFeats = fileData{1};
    classesFiles = fileData{2};
    
    features = []; testFeatures = [];
    data.annotations.classes = [];
    testData.annotations.classes = [];
    for i = 1:length(namePathFeats)
        
        load([input.PATH_DATA input.([phase 'Dataset']).path dataset '_decaf7\' namePathFeats{i} '.mat']);
        % fts = fts ./ repmat(sum(fts,2),1,size(fts,2)); 
        % fts = zscore(fts);
%         randSamples = randperm(size(fts,1));
        % Train portion
%         trainSamples = fts(randSamples(1:min(size(fts,1), numImgs)),:);
        features = [features; fts]; % trainSamples;
        data.annotations.classes = [data.annotations.classes; repmat(classesFiles(i),[size(fts,1), 1])];
        % Test portion
%         if(numImgs < length(randSamples) && (strcmpi(phase,'target') && input.isClassSupervised))
%             testSamples = fts(randSamples(numImgs+1:end),:);
%             testFeatures = [testFeatures; testSamples];
%             testData.annotations.classes = [testData.annotations.classes; repmat(classesFiles(i),[size(testSamples,1), 1])];
%         end        
    end
    input.([phase 'Dataset']).classes = sort_nat(unique(classesFiles))';

    if(isempty(testData.annotations.classes))
        testData = [];
    end
    
    if(input.isZScore)
        feat = [features; testFeatures];
        feat = feat ./ repmat(sum(feat,2),1,size(feat,2)); 
        feat = zscore(feat);
        
        features = feat(1:length(data.annotations.classes),:);
        testFeatures = feat(length(data.annotations.classes)+1:end,:);
    end

end

% if(strcmpi(input.cnnName,'AlexNet'))
%     path = [path 'Alexnet_fc7_' dataset '\'];
% elseif(strcmpi(input.cnnName,'VGG'))
%     path = [path 'VGG16_' dataset '\'];
% else
%     error('[[Caught ERROR: (phase %s - dataset %s) Wrong CNN model specified: %s]]', phase, dataset, input.cnnName);
% end

% Valid for new features (AlexNet and VGG not well organised yet)
function parseLabels(input, dataset, path)

    % Extract all dataset files (needed in all cases)
    folder = dir(path);
    allClassFolders = {folder.name};
    [~, names, exts] = cellfun(@fileparts, allClassFolders, 'UniformOutput', false);
    idxImgPaths = cellfun(@(x) sum(x), cellfun(@(x) strcmp(x, {'.mat'}), exts, 'UniformOutput', false)) == 1;
    sampleNames = names(idxImgPaths)';

    switch(dataset)
        
        case 'awa'
            A = cellfun(@(x) strsplit(x,'_'), sampleNames, 'UniformOutput', false);
            fileID = fopen([input.PATH_DATA input.sourceDataset.path 'labels_awa.txt'],'w');
            for i = 1:length(A)-1
                a = A{i};
                fprintf(fileID,'%s %s\n',sampleNames{i},a{1});
            end
            a = A{end};
            fprintf(fileID,'%s %s',sampleNames{end},a{1});
            fclose(fileID);
        case 'sun'
            fileRead = fopen([input.PATH_DATA input.sourceDataset.path 'sunlist_cat.txt']);
            fileWrite = fopen([input.PATH_DATA input.sourceDataset.path 'labels_sun.txt'],'w');
            tline = fgetl(fileRead);
            while ischar(tline)
                A = strsplit(tline,{'/','.jpg'});
                fprintf(fileWrite,'%s %s\n',A{2},A{1});
                tline = fgetl(fileRead);
            end
            fclose(fileWrite);
            fclose(fileRead);
        case 'msrcorid'
            % 10 repetitions
            list1 = []; list2 = [];
            folder = dir([input.PATH_DATA 'Real\Multi\msrcorid\']);
            [~, names, exts] = cellfun(@fileparts, {folder.name}, 'UniformOutput', false);
            listClasses = names(cellfun(@(x) sum(x), cellfun(@(x) strcmp(x, {''}), exts, 'UniformOutput', false)) == 1);
            for i = 1:length(listClasses)
                folder = dir([input.PATH_DATA 'Real\Multi\msrcorid\' listClasses{i}]);
                [~, imgs, exts] = cellfun(@fileparts, {folder.name}, 'UniformOutput', false);
                listImages = imgs(cellfun(@(x) sum(x), cellfun(@(x) strcmp(x, {'.JPG'}), exts, 'UniformOutput', false)) == 1);
                for j = 1:length(listImages)
                    list1 = [list1; listImages(j)];
                    list2 = [list2; listClasses(i)];
                end
            end
            [list1, idxSorted] = sort_nat(list1);
            list2 = list2(idxSorted);
            checkOk = find(~ismember(sampleNames,list1));
            fileID = fopen([input.sourceDataset.path 'labels_msrcorid.txt'],'w');
            for i = 1:length(list1)
                fprintf(fileID,'%s %s\n',list1{i},list2{i});
            end
            fclose(fileID);
        case 'pascal'
            % multi objects
        case 'caltech256'
            
        case 'mscoco'
        case 'imagenet'
            
    end
    
end

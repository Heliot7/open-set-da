function data = getVisda17(input, dataset, phase)
    
    % Num samples pro class
%     numSamples = 5000;

    % Get data VISDA '17
    dataset_name = 'train';
    if(strcmpi(phase,'target'))
        dataset_name = 'validation';
    end
    path = [input.PATH_DATA dataset.path dataset_name '\'];
    pathFile = [path 'image_list.txt'];
    line_scan = '%s %d';
    file = fopen(pathFile);
    fileData = textscan(file, line_scan);
    fclose(file);
    allNames = strrep(fileData{1},'/','\');
    allMainPaths = repmat({path},[length(allNames), 1]);
    data.imgPaths = strcat(allMainPaths,allNames);
    data.annotations.imgId = (1:length(allNames))';
    % data.annotations.BB = [];
    data.annotations.classes = cell(length(allNames),1);
    classIds = fileData{2};
    for i = 0:length(dataset.classes)-1
        currentClass = repmat(dataset.classes(i+1), sum(classIds == i), 1);
        data.annotations.classes(classIds == i) = currentClass;
    end
    
    % Less samples
%     auxClasses = [];
%     auxPaths = [];
%     for i = 1:length(dataset.classes) 
%         isClass = find(ismember(data.annotations.classes,dataset.classes(i)));
%         isClass = isClass(randperm(length(isClass)));
%         auxClasses = [auxClasses; data.annotations.classes(isClass(1:numSamples))];
%         auxPaths = [auxPaths; data.imgPaths(isClass(1:numSamples))];
%     end
%     data.annotations.classes = auxClasses;
%     data.imgPaths = auxPaths;
    
    % Order based on classes list
%     auxClasses = [];
%     auxPaths = [];
%     for i = 1:length(dataset.classes) 
%         isClass = ismember(data.annotations.classes,dataset.classes(i));
%         auxClasses = [auxClasses; data.annotations.classes(isClass)];
%         isClass = ismember(data.annotations.classes,classes(i));
%         a = data.imgPaths(isClass);
%         auxPaths = [auxPaths; data.imgPaths(isClass)];
%     end
%     data.annotations.classes = auxClasses;
%     data.imgPaths = auxPaths;
    
end

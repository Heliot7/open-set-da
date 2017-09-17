function data = getSyntheticCVC(input, dataset)

    % Get annotations
    annotationsFile = load([input.PATH_DATA dataset.path 'Train\pos\annotation.mat']);
    annotationsFile = annotationsFile.virtual_annotation;

    paths = cell(length(annotationsFile),1);
    data.annotations.imgId = zeros(length(annotationsFile),1);
    data.annotations.BB = zeros(length(annotationsFile),4);
    for i = 1:length(annotationsFile)

        % Get image paths
        paths{i} = [input.PATH_DATA dataset.path 'Train\pos\' annotationsFile(i).filename '.png'];
        data.annotations.imgId(i) = i;
        % Get annotations
        bb = annotationsFile(i).boxes;
        data.annotations.BB(i,:) = [bb.y1, bb.x1, bb.y2 - bb.y1, bb.x2 - bb.x1];
        
    end

    % Get negative paths
    folder = dir([input.PATH_DATA dataset.path 'Train\neg\']);
    classFolders = {folder.name};
    [~, ~, exts] = cellfun(@fileparts, classFolders, 'UniformOutput', false);
    idxImgPaths = cellfun(@(x) sum(x), cellfun(@(x) strcmp(x, {'.png'}), exts, 'UniformOutput', false)) == 1;
    classFolders = classFolders(idxImgPaths)';
    synNegPaths = repmat([input.PATH_DATA dataset.path 'Train\neg\'], length(classFolders), 1);
    negPaths = strcat(synNegPaths, classFolders);
    
    data.imgPaths = [paths; negPaths];
    data.annotations.classes = repmat(dataset.classes(1),[length(data.annotations.imgId) 1]);
        
end

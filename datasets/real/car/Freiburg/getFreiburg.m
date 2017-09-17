function data = getFreiburg(input, dataset)

    % Initilise data
    data = []; data.imgPaths = []; data.annotations.imgId = []; data.annotations.BB = [];
    data.annotations.classes = []; data.annotations.vp.azimuth = [];

    % Annotations
    pathAnnotations = [input.PATH_DATA dataset.path 'annotations\'];
    folder = dir(pathAnnotations);
    modelsAnno = {folder.name};
    [~, ~, exts] = cellfun(@fileparts, modelsAnno, 'UniformOutput', false);
    idxImgPaths = cellfun(@(x) sum(x), cellfun(@(x) strcmp(x, {'.txt'}), exts, 'UniformOutput', false)) == 1;
    modelsAnno = sort_nat(modelsAnno(idxImgPaths)');
    line_scan = '%s %d %d %d %d %d ';
    for i = 1:length(modelsAnno)
        % Open information file:
        file = fopen([pathAnnotations modelsAnno{i}]);
        fileData = textscan(file, line_scan);
        fclose(file);
        % Paths
        absPathAnno = repmat({[input.PATH_DATA dataset.path]}, [length(fileData{1}), 1]);
        paths = cellfun(@strrep, fileData{1}, repmat({'/'},[length(fileData{1}) 1]), repmat({'\'},[length(fileData{1}) 1]), 'UniformOutput', false);
        paths = cellfun(@(x) x(1:end-4), paths, 'UniformOutput', false);
        paths = strcat(absPathAnno, paths, repmat({'.png'}, [length(fileData{1}) 1]));
        data.imgPaths = [data.imgPaths; paths];
        % Ids
        id = length(data.annotations.imgId);
        data.annotations.imgId = [data.annotations.imgId; (id+1:id+length(paths))'];
        % BBs
        bbs = double([fileData{2:5}]);
        data.annotations.BB = [data.annotations.BB; [bbs(:,2) bbs(:,1) bbs(:,4)-bbs(:,2)+1 bbs(:,3)-bbs(:,1)+1]];
        % Classes
        data.annotations.classes = [data.annotations.classes; repmat({'car'},[length(paths) 1])];
        % Azimuths
        az = mod(double(fileData{6}) - 90 + 360 + 20, 360);
        data.annotations.vp.azimuth = [data.annotations.vp.azimuth; az];
    end
    
    % Test Correctness:
%     figure;
%     for i = 356:length(data.imgPaths)
%         img = imread(data.imgPaths{i});
%         BB = data.annotations.BB(i,:);
%         patch = img(BB(1):BB(1)+BB(3)-1,BB(2):BB(2)+BB(4)-1,:);
%         imshow(patch);
%         fprintf('\nimg id %d: %f degrees', i, data.annotations.vp.azimuth(i));
%         keyboard;
%     end
    
end


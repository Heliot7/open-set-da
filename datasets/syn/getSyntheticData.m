function data = getSyntheticData(input, dataset)

    data = defineData();
    for i = 1:length(dataset.classes)
        objClass = dataset.classes{i};
        path = [input.PATH_DATA dataset.path objClass '\' dataset.sub_path];
        folder = dir(path);
        classFolders = {folder.name};
        [~, ~, exts] = cellfun(@fileparts, classFolders, 'UniformOutput', false);
        idxImgPaths = cellfun(@(x) sum(x), cellfun(@(x) strcmp(x, {''}), exts, 'UniformOutput', false)) == 1;
        classFolders = sort_nat(classFolders(idxImgPaths)');
        paths = []; annotations = []; azimuth = []; elevation = []; plane = []; distance = []; kps = [];
        firstPass = true;
        for idxFolder = 1:length(classFolders)
            if(exist([path classFolders{idxFolder} '\annotations.txt'],'file'))
                % -> Only in first pass:
                if(firstPass)
                    file = fopen([path classFolders{idxFolder} '\annotations.txt']);
                    line = squeeze(strsplit(strtrim(fgets(file))));
                    fclose(file);
                    numParts = length(line(10:end)) / 4;
                    firstPass = false;
                end
                [imgNames, imgAnnotations, imgLabels, imgKps, labelsKps] = ...
                    getSyntheticAnnotations([path classFolders{idxFolder} '\annotations.txt'], numParts);
                % - Add absolute file paths
                paths = [paths; strcat(repmat({[path classFolders{idxFolder} '\']}, length(imgNames), 1), imgNames)];
                % - Add annotations
                annotations = [annotations; imgAnnotations];
                % - Add labels
                azimuth = [azimuth; imgLabels{1}];
                elevation = [elevation; imgLabels{2}];
                plane = [plane; imgLabels{3}];
                distance = [distance; imgLabels{4}];
                kps = [kps; imgKps];
            end
        end

%         [imgPaths, ~, idxsUnique] = unique(paths);
%         [imgPaths, idxsNat] = sort_nat(imgPaths);
%         imgId = zeros(length(idxsNat),1);
%         for id = 1:length(idxsNat)
%             imgId(idxsUnique == idxsNat(id)) = id;
%         end
        data.imgPaths = [data.imgPaths; paths];
        data.annotations.imgId = [data.annotations.imgId; length(data.annotations.imgId)+(1:length(paths))'];
        data.annotations.BB = [data.annotations.BB; annotations];
        data.annotations.classes = [data.annotations.classes; repmat(dataset.classes(i),[length(paths) 1])];
        data.annotations.vp.azimuth = [data.annotations.vp.azimuth; azimuth];
        data.annotations.vp.elevation = [data.annotations.vp.elevation; elevation];
        data.annotations.vp.plane = [data.annotations.vp.plane; plane];
        data.annotations.vp.distance = [data.annotations.vp.distance; distance];
        data.annotations.parts = [data.annotations.parts; kps];
        data.partLabels = [data.partLabels; {labelsKps}];
    end
    
    % Info files:
    for i = 1:length(dataset.classes)
        class = dataset.classes{i};
        idxs = ismember(data.annotations.classes, class);
        bb = data.annotations.BB(idxs,:);
        num = size(bb,1);
        row = [mean(bb(:,3)) std(bb(:,3))];
        col = [mean(bb(:,4)) std(bb(:,4))];
        fprintf('[%s]\nTRAIN %d samples, row: %.1f,%.1f col: %.1f,%.1f\n', class, num, row, col);
    end
%     keyboard;
    
end

% Get annotations of all grid models:
function [paths, annotations, labels, kps, labelsKps] = getSyntheticAnnotations(path, numParts)

    line_scan = ['%s %f %f %f %f %f %f %f %f ' repmat('%s %f %f %f ', [1 numParts])];
    file = fopen(path);
    fileData = textscan(file, line_scan);
    fclose(file);
    paths = fileData{1};
    annotations = double([fileData{2:5}]); % [ROW, COL, HEIGHT, WIDTH]
    annotations(:,1:2) = annotations(:,1:2) + 1;
    labels = fileData(6:9);
    raw_kps = fileData(10:end);
    kp = [];
    labelsKps = cell(1,numParts);
    for idxKp = 1:numParts
        names = raw_kps{(idxKp-1)*4 + 1};
        labelsKps(idxKp) = names(1);
        kpRow = raw_kps{(idxKp-1)*4 + 2};
        kpCol = raw_kps{(idxKp-1)*4 + 3};
        isVis = raw_kps{(idxKp-1)*4 + 4};
        isVis(isVis < 0 ) = 0; % No distinction between occluded and truncated
        kpCol(isVis == 0) = 0;
        kpRow(isVis == 0) = 0;
        kp = [kp; [kpRow, kpCol, isVis]];
    end
    numSamples = length(paths);        
    kps = cell(numSamples,1);
    for idxKp = 1:numSamples
        kps{idxKp}= kp(idxKp:numSamples:end,:);
    end
    badRow = find(annotations(:,3) < 16);
    badCol = find(annotations(:,4) < 16);
    badSamples = unique([badRow; badCol]);
%     if(~isempty(badSamples))
%         for i = 1:length(badSamples)
%             annotations(badSamples(i),:)
%         end
%         keyboard;
%     end
    paths(badSamples) = [];
    annotations(badSamples,:) = [];
    kps(badSamples) = [];
    for i = 1:4
        aux_lab = labels{i};
        aux_lab(badSamples) = [];
        labels{i} = aux_lab;
    end    
end


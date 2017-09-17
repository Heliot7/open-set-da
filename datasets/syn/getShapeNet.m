function data = getShapeNet(input, dataset)

    nameDataset = class(input.targetDataset);
    % nameDataset = 'Pascal3D'; % Use this generation for O3D as well...
    
    data.imgPaths = [];
    data.annotations.imgId = []; data.annotations.BB = [];
    data.annotations.classes = []; data.annotations.vp = [];
    data.annotations.vp.azimuth = []; data.annotations.vp.elevation = [];
    data.annotations.vp.plane = []; data.annotations.vp.distance = [];
    for i = 1:length(dataset.classes)
        id = 1;
        path = [input.PATH_DATA dataset.path nameDataset '\' dataset.synsets{i} '\'];
        folder = dir(path);
        modelFolders = {folder.name};
        [~, ~, exts] = cellfun(@fileparts, modelFolders, 'UniformOutput', false);
        idxModelPaths = cellfun(@(x) sum(x), cellfun(@(x) strcmp(x, {''}), exts, 'UniformOutput', false)) == 1;
        modelFolders = sort_nat(modelFolders(idxModelPaths)');
        paths = cell(250000,1);
        for idxFolder = 1:length(modelFolders)
            pathFolder = [path modelFolders{idxFolder} '\'];
            folder = dir(pathFolder);
            list_images = {folder.name};
            [~, ~, exts] = cellfun(@fileparts, list_images, 'UniformOutput', false);
            % If empty, check jpg!
            idx_images = cellfun(@(x) sum(x), cellfun(@(x) strcmp(x, {'.png','.jpg'}), exts, 'UniformOutput', false)) == 1;
            list_images = list_images(idx_images)';
            % - Add absolute file paths
            % paths = [paths; strcat(repmat({pathFolder}, length(list_images), 1), list_images)];
            paths(id:id+length(list_images)-1) = strcat(repmat({pathFolder}, length(list_images), 1), list_images);
            id = id + length(list_images);
            if(mod(idxFolder,1000) == 0)
                fprintf('[%s] Reading folder %d/%d...\n', dataset.classes{i}, idxFolder, length(modelFolders));
            end
        end
        paths = paths(1:id-1);
        numSamples = length(paths);
        % Shuffle all paths to get mixes models
        % (not necessary, done in lmdb)
        % perms = randperm(numSamples);
        % paths = paths(perms);
        bb = zeros(numSamples,4); azimuth = zeros(numSamples,1);
        elevation = zeros(numSamples,1); plane = zeros(numSamples,1); distance = zeros(numSamples,1);
        for idxImg = 1:min(250000,length(paths)) % 250000
            img_name = paths{idxImg};
            split_img = strsplit(img_name,'_');
            % - Add viewpoint data
            az = split_img{3};
            azimuth(idxImg) = mod(str2num(az(2:end)), 360); % [0..359]
            el = split_img{4};
            elevation(idxImg) = str2num(el(2:end));
            pl = split_img{5};
            plane(idxImg) = str2num(pl(2:end));
            if(strcmpi(nameDataset,'ObjectNet3D'))
                plane(idxImg) = -1*plane(idxImg);
            end
            % correct plane: temporal
            pl = plane(idxImg);
            if(pl >= 180)
                pl = pl - 360;
            end
            plane(idxImg) = pl;
            dist = split_img{6};
            distance(idxImg) = str2num(dist(2:end-4));
            % - Add BB (whole image -> 1 1 w h)
            info = imfinfo(img_name);
            bb(idxImg,:) = [1 1 info.Height info.Width];
            if(mod(idxImg,10000) == 0)
                fprintf('[%s] Reading sample %d/%d...\n', dataset.classes{i}, idxImg, numSamples);
            end
        end
        data.imgPaths = [data.imgPaths; paths];
        data.annotations.imgId = [data.annotations.imgId; length(data.annotations.imgId)+(1:idxImg)'];
        data.annotations.BB = [data.annotations.BB; bb(1:idxImg,:)];
        data.annotations.classes = [data.annotations.classes; repmat(dataset.classes(i),[idxImg 1])];
        data.annotations.vp.azimuth = [data.annotations.vp.azimuth; azimuth(1:idxImg)];
        data.annotations.vp.elevation = [data.annotations.vp.elevation; elevation(1:idxImg)];
        data.annotations.vp.plane = [data.annotations.vp.plane; plane(1:idxImg)];
        data.annotations.vp.distance = [data.annotations.vp.distance; distance(1:idxImg)];
    end
    
end



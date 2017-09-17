function [data, testData] = getObjCat3D(input, dataset)

    data.imgPaths = [];
    data.annotations.imgId = []; data.annotations.BB = [];
    data.annotations.classes = []; data.annotations.vp.azimuth = [];
    testData.imgPaths = [];
    testData.annotations.imgId = []; testData.annotations.BB = [];
    testData.annotations.classes = []; testData.annotations.vp.azimuth = [];
    for idxClasses = 1:length(dataset.classes)

        className = dataset.classes{idxClasses};
        path = [input.PATH_DATA dataset.path className '\'];
        folder = dir(path);
        classFolders = {folder.name};
        [~, ~, exts] = cellfun(@fileparts, classFolders, 'UniformOutput', false);
        idxFolders = cellfun(@(x) sum(x), cellfun(@(x) strcmp(x, {''}), exts, 'UniformOutput', false)) == 1;
        classFolders = sort_nat(classFolders(idxFolders));
        
        fullPaths = []; fullAnnotations = [];
        fullLabelsAzimuth = []; fullLabelsElevation = []; fullLabelsDistance = [];
        % classFolders(strcmpi(classFolders,'mat_features')) = [];
        for idxFolder = 1:length(classFolders)

            % From 1 till 7 seq. -> train. From 8 till 10 seq. -> test part
            if(idxFolder == 1 || idxFolder == 8)
                fullPaths = []; fullAnnotations = [];
                fullLabelsAzimuth = []; fullLabelsElevation = []; fullLabelsDistance = [];
            end

            % Get paths without any order
            folderClass = dir([path classFolders{idxFolder} '\']);
            folderClass = {folderClass.name};
            [~, ~, exts] = cellfun(@fileparts, folderClass, 'UniformOutput', false);
            imgPaths = cellfun(@(x) sum(x), cellfun(@(x) strcmp(x, {'.png', '.jpg', '.pgm', '.bmp'}), exts, 'UniformOutput', false)) == 1;
            imgPaths = folderClass(imgPaths)';
            paths = repmat([path classFolders{idxFolder} '\'], size(imgPaths,1), 1);
            paths = strcat(paths, imgPaths);

            % Parse the correspondent class: [azimuth, elevation, distance]
            if(strcmpi(className,'car'))
                if(idxFolder < 7)
                    regstr = sprintf('\\\\%s_A(\\d+)_H(\\d+)_S(\\d+)%s',className,'.bmp');
                else
                    regstr = sprintf('\\\\%s_A(\\d+)_H(\\d+)_S(\\d+)%s',[className num2str(idxFolder)],'.bmp');
                end
                listAz = [180 135 90 45 0 315 270 225];
            elseif(strcmpi(className,'bicycle'))
                if(idxFolder < 10)
                    regstr = sprintf('\\\\%s_A(\\d+)_H(\\d+)_S(\\d+)%s',[className '_0' num2str(idxFolder)],'.bmp');
                else
                    regstr = sprintf('\\\\%s_A(\\d+)_H(\\d+)_S(\\d+)%s',[className '_' num2str(idxFolder)],'.bmp');
                end
                listAz = [180 225 270 315 0 45 90 135 ];
            end
            A_E_D = regexpi(paths', regstr, 'tokens')';

            labelsAzimuth = zeros(length(paths), 1);
            labelsElevation = zeros(length(paths), 1);
            labelsDistance = zeros(length(paths), 1);
            annotations = zeros(length(paths), 4);
            for idxImg = 1:length(imgPaths)

                % - Assign annotations per img
                relPath = imgPaths{idxImg};
                mask = readMask([path classFolders{idxFolder} '\mask\' relPath(1:end-4) '.mask']);
                row0 = find(sum(mask,2), 1 );
                row1 = find(sum(mask,2), 1, 'last' );
                col0 = find(sum(mask,1), 1 );
                col1 = find(sum(mask,1), 1, 'last' );
                annotations(idxImg,:) = [row0, col0, row1-row0, col1-col0];

                % - Assign labels per img
                % -> AZIMUTH
                labelsAzimuth(idxImg) = listAz(str2double(A_E_D{idxImg}{1}{1}));

                % -> ELEVATION
                elevation = str2double(A_E_D{idxImg}{1}{2});
                if(elevation == 1)
                    labelsElevation(idxImg) = 0;
                elseif(elevation == 2)
                    labelsElevation(idxImg) = 30;
                end

                % -> DISTANCE
                labelsDistance(idxImg) = str2double(A_E_D{idxImg}{1}{3});

            end

            fullPaths = [fullPaths; paths];
            fullAnnotations = [fullAnnotations; annotations];
            fullLabelsAzimuth = [fullLabelsAzimuth; labelsAzimuth];
            fullLabelsElevation = [fullLabelsElevation; labelsElevation];
            fullLabelsDistance = [fullLabelsDistance; labelsDistance];

            if(idxFolder == 7) % train part
                data.imgPaths = [data.imgPaths; fullPaths];
                minId = length(data.annotations.imgId);
                data.annotations.imgId = [data.annotations.imgId; (minId+1:minId+length(fullPaths))'];
                data.annotations.BB = [data.annotations.BB; fullAnnotations];
                data.annotations.classes = [data.annotations.classes; repmat(dataset.classes(idxClasses),[length(fullPaths) 1])];
                data.annotations.vp.azimuth = [data.annotations.vp.azimuth; fullLabelsAzimuth];
            elseif(idxFolder == 10) % test part
                testData.imgPaths = [testData.imgPaths; fullPaths];
                minId = length(testData.annotations.imgId);
                testData.annotations.imgId = [testData.annotations.imgId; (minId+1:minId+length(fullPaths))'];
                testData.annotations.BB = [testData.annotations.BB; fullAnnotations];
                testData.annotations.classes = [testData.annotations.classes; repmat(dataset.classes(idxClasses),[length(fullPaths) 1])];
                testData.annotations.vp.azimuth = [testData.annotations.vp.azimuth; fullLabelsAzimuth];
            end
        end

    end

end

function m = readMask(maskfile)

    fid = fopen(maskfile,'r');
    h = fscanf(fid,'%d',1);
    w = fscanf(fid,'%d',1);
    m = fscanf(fid,'%d',w*h);
    m = reshape(m,[h w]);    
    fclose(fid);
    
end



function comparisonVideo(outputFolder, outputName, folderTop, tagTop, folderDown, tagDown, frameRate)

    if(nargin < 7)
        frameRate  = 3;
    end

    folder = dir(folderTop);
    folders = {folder.name};
    [~, strFolders, ~] = cellfun(@fileparts, folders, 'UniformOutput', false);
    idxFolders = strfind(strFolders, tagTop);
    strFolders = sort_nat(strFolders(~cell2mat(cellfun(@isempty, idxFolders, 'UniformOutput', false))));
    pathsTop = strcat({[folderTop '\']}, strFolders', {'.png'});

    folder = dir(folderDown);
    folders = {folder.name};
    [~, strFolders, ~] = cellfun(@fileparts, folders, 'UniformOutput', false);
    idxFolders = strfind(strFolders, tagDown);
    strFolders = sort_nat(strFolders(~cell2mat(cellfun(@isempty, idxFolders, 'UniformOutput', false))));
    pathsDown = strcat({[folderDown '\']}, strFolders', {'.png'});
    
    % Initialise video
    video = VideoWriter([outputFolder '\' outputName '.avi']);
    video.FrameRate = frameRate;
    video.Quality = 100;
    open(video);
    
    % Create combi of sequences top/down
    fprintf('Creating combo video:\n');
    for i = 1:min(length(pathsTop),length(pathsDown))
        fprintf('image %d/%d\n', i, min(length(pathsTop),length(pathsDown)));
        % Read images
        imgTop = imread(pathsTop{i});
        imgDown = imread(pathsDown{i});
        % Crop white parts
        sizeOneImg = 788-114+1;
        img = zeros(2*sizeOneImg+25,size(imgTop,2),3,'uint8');
        % Copy into new image matrix
        img(1:sizeOneImg,:,:) = imgTop(114:788,:,:);
        img(sizeOneImg+26:end,:,:) = imgDown(114:788,:,:);
        % Store in the video
        writeVideo(video, img);
    end
    
    close(video);

end


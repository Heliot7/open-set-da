function saveVideoDetection(outFolder, outName, inFolder, inName, frameRate)

    if(nargin < 5)
        frameRate  = 3;
    end

    folder = dir(inFolder);
    folders = {folder.name};
    [~, strFolders, ~] = cellfun(@fileparts, folders, 'UniformOutput', false);
    idxFolders = strfind(strFolders, inName);
    strFolders = sort_nat(strFolders(~cell2mat(cellfun(@isempty, idxFolders, 'UniformOutput', false))));
    paths = strcat({[inFolder '\']}, strFolders', {'.png'});

    % Initialise video
    video = VideoWriter([outFolder '\' outName '.avi']);
    video.FrameRate = frameRate;
    video.Quality = 100;
    open(video);
    
    for idxImage = 1:length(paths)
        img = imread(paths{idxImage});
        writeVideo(video, img);
    end
    
    close(video);
    
end



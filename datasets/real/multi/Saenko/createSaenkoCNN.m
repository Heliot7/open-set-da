function createSaenkoCNN()

    numClasses = 31;
    idCNN = 'FC7_FT';
    isFineTuning = true;
    
    input = InputParameters;
    path = [input.PATH_DATA 'Real\DomainAdaptation\Saenko\'];
    datasets = {'amazon', 'caltech', 'dslr', 'webcam'};
    objects = {'00 back_pack', '01 bike', '02 calculator', '03 headphones', '04 keyboard', '05 laptop_computer', ...
        '06 monitor', '07 mouse', '08 mug', '09 projector'};
    if(numClasses == 31)
        datasets = {'amazon', 'dslr', 'webcam'};
        objects = {'00 back_pack', '01 bike', '02 calculator', '03 headphones', '04 keyboard', '05 laptop_computer', ...
            '06 monitor', '07 mouse', '08 mug', '09 projector', '10 bike_helmet', '11 bookcase', '12 bottle', '13 desk_chair', ...
            '14 desk_lamp', '15 desktop_computer', '16 file_cabinet', '17 letter_tray', '18 mobile_phone', '19 paper_notebook' ...
            '20 pen', '21 phone', '22 printer', '23 punchers', '24 ring_binder', '25 ruler', '26 scissors', '27 speaker', '28 stapler', ...
            '29 tape_dispenser', '30 trash_can'};
    end
    
    numDims = 4096;
    input.cnnName = 'FT_AlexNet_DA';    
    for dom = 1:length(datasets)
        CNN_Caffe('unloadNet');
        CNN_Caffe('loadNet', input.cnnName, [input.cnnName '_' datasets{dom}], strrep(input.PATH_CNN, '\', '/'));
        for dat = 1:length(datasets)
            fprintf('Extracting features with domain %s for dataset %s\n', datasets{dom}, datasets{dat});
            fts = []; labels = [];
            for o = 1:length(objects)
                fprintf('Accessing %s object folder\n', objects{o});
                obj_path = [path datasets{dat} '\' objects{o} '\'];
                folder = dir(obj_path);
                classFolders = {folder.name};
                [~, ~, exts] = cellfun(@fileparts, classFolders, 'UniformOutput', false);
                idxImgPaths = cellfun(@(x) sum(x), cellfun(@(x) strcmp(x, {'.jpg'}), exts, 'UniformOutput', false)) == 1;
                jpgImgs = classFolders(idxImgPaths)';
                obj_path = repmat({obj_path}, length(jpgImgs), 1);
                obj_path = strcat(obj_path, jpgImgs);
                featCNN = zeros(length(obj_path),numDims);    
                for idxImg = 1:length(obj_path)
                    fprintf('Computing CNN feat. of image %d (class %d) \n', idxImg, o);
                    img = imread(obj_path{idxImg});
                    img = grey2rgb(img);
                    img = img(:, : , [3, 2, 1]);
                    img = permute(img, [2, 1, 3]);
                    img = imresize(img, [227 227], 'bilinear');
                    imgB = img(:,:,1); imgG = img(:,:,2); imgR = img(:,:,3);
                    means = [mean(imgB(:)),mean(imgG(:)),mean(imgR(:))];
                    img = single(img);
                    img = bsxfun(@minus, img, reshape(means,1,1,3));
                    f = CNN_Caffe('getFeatures', img, 'fc7');
                    featCNN(idxImg,:) = f;
                end
                fts = [fts; featCNN];
                labels = [labels; o*ones(length(obj_path),1)];
            end
            save_path = [path 'office_fc7_FT\' input.cnnName '-' datasets{dom} '_' datasets{dat} '_CNN-' idCNN '_L' num2str(numClasses)]; 
            save(save_path, 'fts', 'labels', '-v7.3');
        end
    end    
end

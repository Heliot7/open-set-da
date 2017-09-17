function saveImagesDetection(imgName, input, testData, metaData, BBs, detIds, scores, detections)
    
    % Normalise all scores to show strength of score in BB's colour
    if(~isempty(scores))
        strength = scores ./ max(scores);
    else
        strength = 1.0;
    end

    mainColour = [0.5 0 0];
    % No gt provided
    if(isempty(detections))
        mainColour = [0.5 0.75 1];
    end

    f = figure;
    set(f, 'visible', 'off');
    numImages = min(input.numImages, length(testData.imgPaths));
    for idxImage = 1:numImages

        img = imread(testData.imgPaths{idxImage});
        imshow(img);
        set(gca,'position',[0 0 1 1],'units','normalized');
        axis off;
        
        % Draw BB of ignored portions of image, if existing
        if(isfield(testData, 'ignore'))
            listIgnores = testData.ignore.BB(testData.ignore.imgId == idxImage,:);
            for idxAnno = 1:size(listIgnores,1)
                listXYWH = [listIgnores(idxAnno,2:-1:1), listIgnores(idxAnno,4:-1:3)];
                rectangle('position', listXYWH, 'LineWidth', 1, 'EdgeColor', [0.2 0.2 0.2]);
            end
        end
        
        % Draw BB of grount truth detection
        if(isfield(testData,'annotations') && ~isempty(testData.annotations.BB))        
            listAnnotations = testData.annotations.BB(testData.annotations.imgId == idxImage,:);
            for idxAnno = 1:size(listAnnotations,1)
                listXYWH = [listAnnotations(idxAnno,2:-1:1), listAnnotations(idxAnno,4:-1:3)];
                rectangle('position', listXYWH, 'LineWidth', 1, 'EdgeColor', [0 0 1]);
            end
        end
        
        listBBs = [];
        if(~isempty(BBs))
            listBBs = BBs(BBs(:,5) == idxImage, 1:4);
            if(~isempty(listBBs))
                listScores = scores(BBs(:,5) == idxImage);
                listLabels = detIds(BBs(:,5) == idxImage,:);
                listDetections = detections(BBs(:,5) == idxImage);
            end
        end
        
        % Draw BB + info of all detections in the image
        hold on;
        for idx = 1:size(listBBs,1)

            % BB-Colour: Whether there is matching
            colour = mainColour + [0.5*strength(idx) 0 0];
            if(isfield(testData,'annotations') && ~isempty(testData.annotations.BB))
                if(listDetections(idx) == 1)
                    colour = [0 0.5+0.5*strength(idx) 0];
                elseif(listDetections(idx) == -1)
                    colour = [0.9 0.9 0.9];
                end
            else
                colour = [0 0.8 0];
            end

            bb = listBBs(idx,:);
            listXYWH = [bb(2:-1:1), bb(4:-1:3)];
            rectangle('position', listXYWH, 'LineWidth', 1, 'EdgeColor', colour);
            str = sprintf('[%3.2f]', listScores(idx));
            for idxMeta = 1:length(metaData)
                % field = metaData{idxMeta};
                dataStr = sprintf(' %s', listLabels{idx, idxMeta});
                str = strcat(str, dataStr);
                if(input.isShowPose && strcmpi(metaData{idxMeta},'azimuth'))
                    rectangle('position', [bb(2:-1:1), bb(4:-1:3)], 'LineWidth', 1, 'EdgeColor', [0.9 0.7 0.7]);
                    hold on;
                    angle = str2num(listLabels{idx, idxMeta});
                    sizeArrow = min(bb(3:4)/2);
                    quiver(bb(2)+bb(4)/2,bb(1)+bb(3)/2,sin(angle*pi/180-pi)*sizeArrow,cos(angle*pi/180)*sizeArrow,...
                        'Color','y','LineWidth',2,'MaxHeadSize',3);
                end                
                
            end
            text(bb(2), bb(1)-10, str, 'Color', colour, 'FontSize', 6);

        end
        hold off;

        createDir([getResultsPath(input) '/detOutput']);
        saveas(f, [getResultsPath(input) '/detOutput/' imgName num2str(idxImage) '.png']);
        
    end

end


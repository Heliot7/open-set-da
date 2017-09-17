function saveImagesLabelTransfer(input, metaData, data, newLabels, gt, scores)

    f = figure;
    set(f, 'visible', 'off');
    scores = max(scores,[],2);
    idx = 0; previousPath = '';
    for idxImg = 1:length(data.imgPaths)

        fprintf('Save image with annotated labels %d/%d\n', idxImg, length(data.imgPaths));
        if(~strcmpi(previousPath, data.imgPaths{idxImg}))
            idx = idx + 1;
            previousPath = data.imgPaths{idxImg};
            img = imread(data.imgPaths{idxImg});
            imshow(img);
            set(gca,'position',[0 0 1 1],'units','normalized');
            axis off;
        end
        
        % Get all BBs 
        isImg = find(data.annotations.imgId == idxImg);
        annotation = num2cell(data.annotations.BB(isImg,:));
        [row, col, height, width] = annotation{:};
        
        % Draw BB of selected sample in the image
        for i = 1:length(row)
            colour = [1 0.25 0.25];
            if(sum(ismember(gt(isImg(i),:),newLabels(isImg(i),:))) == size(newLabels,2))
                colour = [0.25 1 0.25];
            end
            hold on;
            rectangle('position',[col row width height], 'LineWidth', 1, 'EdgeColor', colour);
            str = sprintf('s: %3.2f', scores(idxImg));
            for idxMeta = 1:length(metaData)
                field = metaData{idxMeta};
                value = newLabels{isImg,idxMeta};
                if(~ischar(value))
                    value = num2str(value);
                end
                dataStr = sprintf(' %s: %s', field(1), value);
                str = strcat(str, dataStr);
                if(input.isShowPose && strcmpi(field,'azimuth'))
                    bb = data.annotations.BB(isImg,:);
                    rectangle('position', [bb(2:-1:1), bb(4:-1:3)], 'LineWidth', 1, 'EdgeColor', [0.9 0.7 0.7]);
                    angle = str2num(value);
                    sizeArrow = min(bb(3:4)/2);
                    quiver(bb(2)+bb(4)/2,bb(1)+bb(3)/2,sin(angle*pi/180-pi)*sizeArrow,cos(angle*pi/180)*sizeArrow,...
                        'Color','y','LineWidth',2,'MaxHeadSize',3);
                end
            end
            text(col, row-5, str, 'Color', colour, 'FontSize', 8);
            hold off;
        end

        % Save image if finished or next sample is in another one
        if(idxImg == length(data.imgPaths) || ~strcmpi(data.imgPaths{idxImg}, data.imgPaths{idxImg+1}))
            createDir([getResultsPath(input) '/transferOutput']);
            saveas(f, [getResultsPath(input) '/transferOutput/img' num2str(idx) '.png']);
        end
    end

end


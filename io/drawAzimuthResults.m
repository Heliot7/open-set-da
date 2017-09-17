function drawAzimuthResults(labels, strClass, gtLabels, probs, fileName)

    % Shift labels to the left side
%     labels = mod(labels - (360.0 / length(labels) / 2.0) + 360.0, 360);
%     % Assign rounded value to the gt angles
%     auxLabels = gtLabels;
%     for i = 1:length(labels)
%         gtLabels(auxLabels == i) = mod(labels(i) + 360, 360);
%     end

    f = figure;
    set(f, 'visible', 'off');
    set(f, 'Position', [500, 100, 800, 800]);
    title('Viewpoint accuracy - Probabilities');
    axis square;
    axis off;
    hold on;
    % Draw portions each 
    drawPortions(labels, gtLabels, probs);
    % Draw angle separation lines
    drawLines(labels);
    axis([-1.75 1.75 -1.75 1.75]);
    % Draw % of accuracy around viewpoints
    drawText (labels, probs);
    caxis([0 1]);
    h = colorbar;
    set(h, 'ylim', [0 1]);  
    hold off;
    
    fileName = [fileName ' viewchart ' strClass(1:3) '.png'];
    if(exist(fileName,'file'))
        fileName = [fileName(1:end-4) ' 2.png'];
    end
    saveas(f, fileName);
    
end

function drawPortions(angles, gtLabels, probs)

    [~, gtLabels] = getAzimuthId(angles, gtLabels);
    angles = mod(angles + 90.0, 360); % visualisation offset
    gtLabels = mod(gtLabels + 90.0, 360); % for gt as well
    maxRep = max(histc(gtLabels,unique(gtLabels)));
    for i = 1:length(angles)
        currAngle = mod(angles(i) - 360, 360)*pi/180.0 - (2*pi / length(angles) / 2.0);
        if(i == length(angles))
            ni = 1;
        else 
            ni = i + 1;
        end
        if(angles(ni) < angles(i))
            nextAngle = (angles(ni)+360)*pi/180.0 - (2*pi / length(angles) / 2.0);
            rad = sum(gtLabels == angles(i)) / maxRep;
        else
            nextAngle = mod(angles(ni)-360, 360)*pi/180.0 - (2*pi / length(angles) / 2.0);
            rad = sum(gtLabels == angles(i)) / maxRep;
        end
        
        portion = rad.*[[-cos(currAngle:0.05:nextAngle) -cos(nextAngle)]; [sin(currAngle:0.05:nextAngle) sin(nextAngle)]];
        cmap = colormap;
        colour = round(probs(i)*size(cmap,1));
        if(colour <= 0)
            colour = 1;
        end
        fill([0,portion(1,:),0],[0,portion(2,:),0], cmap(colour,:)); %, 'LineSmoothing', 'on');
    end
    
    % Number of sample representation - circles
    angles = 0:0.05:2*3.14159;
    xAngles = cos([angles 2*pi]);
    yAngles = sin([angles 2*pi]);
    plot(xAngles,yAngles, 'LineSmoothing', 'on', 'LineWidth', 1.1, 'Color', [0,0,0]);
    plot(0.5*xAngles,0.5*yAngles, 'LineSmoothing', 'on', 'LineWidth', 1.1, 'Color', [0,0,0]);
    text(cos(45), sin(45)-0.1, mat2str(maxRep),  'FontSize', 7, 'HorizontalAlignment', 'center');
    text(cos(45)*0.4, sin(45)*0.4, mat2str(round(maxRep/2.0)), 'FontSize', 7, 'HorizontalAlignment', 'center');
end

function drawLines(angles)

    angles = mod(angles - (360.0 / length(angles) / 2.0) + 360.0, 360);
    view4Angles = [315, 135, 45, 225];
    for i = 1:length(view4Angles)
        if(i <= 2)
            selAngle = min(angles(angles >= view4Angles(i)));
        else
            selAngle = max(angles(angles <= view4Angles(i)));
        end
        yLine = sin((angles(angles == selAngle)+90)*pi/180.0)*1.1;
        xLine = -cos((angles(angles == selAngle)+90)*pi/180.0)*1.1;
        plot([0 xLine], [0 yLine], 'LineWidth', 2, 'Color', [0,0,0], 'LineSmoothing', 'on');
    end
    quiver(0, 0, 0, 1.11, 'LineWidth', 1.5, 'Color', [0,0,0], 'AutoScale', 'off');

end

function drawText(angles, probs)

    if(length(angles) <= 36)
        angles = mod(angles + 90.0, 360); % visualisation offset
        pos = [angles(2:end); angles(1)] - 360/length(angles);
        yPos = sin(pos*pi/180.0)*1.2;
        xPos = -cos(pos*pi/180.0)*1.2;
        for i = 1:length(pos)
            strProb = sprintf('%0.2f', probs(i));            
            text(xPos(i) - 0.1, yPos(i), strProb, 'FontSize', 9);
        end
    end
        
    % Draw labels front/back/left/right and arrow
    text(0, 1.5, 'front', 'HorizontalAlignment', 'center', 'FontSize', 12);
    text(0,-1.5, 'back', 'HorizontalAlignment', 'center', 'FontSize', 12);
    text(-1.5,0, 'left', 'HorizontalAlignment', 'center', 'FontSize', 12);
    text(1.5,0, 'right', 'HorizontalAlignment', 'center', 'FontSize', 12);

end


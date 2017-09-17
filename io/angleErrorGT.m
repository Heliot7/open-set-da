function [angleError, stdDevAngle] = angleErrorGT(gtLabels, assignedLabels, fileName, strClass, is4ViewSupervised)

    f = figure;
    set(f, 'visible', 'off');
    set(f, 'Position', [500, 100, 800, 800]);

    % Compute Distances (sorted ascend)
    dist = sort(min(abs(gtLabels - assignedLabels), abs(abs(gtLabels - assignedLabels) - 360)));
    angleError = mean(dist);
    stdDevAngle = std(dist);

    % Cumulative histogram
    cumDist = dist;
    cumNum = (1:length(cumDist))'/length(cumDist)*100;
    
    % Max visualisation distances
    if(is4ViewSupervised)
        maxAngleError = 90;
    else
        maxAngleError = 180;
    end
    
    cumDist = [0.0; cumDist; maxAngleError];
    cumNum = [0.0; cumNum; 100.0];
    plot(cumDist, cumNum, 'Color', [0.5,0.5,0.5], 'LineWidth', 3, 'LineSmoothing', 'off');
    hold on;
%     area(cumDist, cumNum, 'FaceColor', [0.92,0.92,0.92]);

    xlim([0 maxAngleError]);
    ylim([0 100]);
    title(sprintf('Cumulative Histogram - Average Angle Error: %.2f° with Std Dev %.2fº', angleError, stdDevAngle));
    xlabel('Degrees of angle deviation');
    ylabel('Cumulative % of correct viewpoint classifications');

    % Storage Correct Cumulative Histogram
    strAngle = sprintf('%.2f', angleError);
    strStd = sprintf('%.2f', stdDevAngle);
    pathCorrectName = [fileName ' AE ' strAngle ' ' strStd ' ' strClass(1:3) '.pdf'];
    if(exist(pathCorrectName,'file'))
        pathCorrectName = [pathCorrectName(1:end-4) ' 2.pdf'];
    end
    saveas(f, pathCorrectName);

end


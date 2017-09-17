function [angleError, stdDevAngle] = angleError(labels, elemMat, fileName, is4ViewSupervised, typeAccuracy)

    labels = sort(labels);
    f = figure;
    set(f, 'visible', 'off');
    set(f, 'Position', [500, 100, 800, 800]);

    distMat = repmat(1:length(labels),[length(labels) 1]);
    dist = mod(distMat - distMat', length(labels));
    dist2 = mod(distMat' - distMat, length(labels));
    dist = min(dist, dist2);
    maxDist = max(max(dist));
    allElem = sum(sum(elemMat));
    
    if(strcmpi(typeAccuracy,'elem'))
        data = zeros(maxDist, 1);
        for i = 1:maxDist
            elems = elemMat(dist == i);
            data(i) = sum(elems);
        end
        % - Old Angle error with rounded fined angles
        % Avg angle error
        angleError = sum(data./allElem.*labels(2:maxDist+1));
        % Std Dev angle error
        diff = abs(labels(2:maxDist+1)) - repmat(angleError, [maxDist 1]);
        stdDevAngle = sqrt(sum(data.*(diff.*diff))./ allElem);
        correctLabels = allElem-sum(data);
    elseif(strcmpi(typeAccuracy,'prob'))
        aux_mean = zeros(length(labels),1);
        aux_std = zeros(length(labels),1);
        aux_data = zeros(length(labels), maxDist);
        for idxView = 1:length(labels)
            viewMat = elemMat(idxView,:);
            for idxRes = 1:maxDist
                elems = viewMat(dist(idxView,:) == idxRes);
                aux_data(idxView,idxRes) = sum(elems);
            end
            aux_mean(idxView) = sum(aux_data(idxView,:) ./ sum(viewMat) .* labels(2:maxDist+1)');
            diff = abs(labels(2:maxDist+1)) - repmat(aux_mean(idxView), [maxDist 1]);
            aux_std(idxView) = sqrt(sum(aux_data(idxView,:)'.*(diff.*diff))./ sum(viewMat));
        end
        angleError = mean(aux_mean);
        stdDevAngle = mean(aux_std);
        aux_data = aux_data ./ repmat(sum(elemMat,2),[1 size(aux_data,2)]);
        data = mean(aux_data)';
        correctLabels = 1-sum(data);
    end
    % Cumulative histogram
    for i = 2:length(data)
        data(i) = data(i) + data(i-1);
    end
    
    if(strcmpi(typeAccuracy,'elem'))
        dataCorrect = [correctLabels; correctLabels + data];
        data = data ./ allElem;
        dataCorrect = dataCorrect ./ allElem;
    elseif(strcmpi(typeAccuracy,'prob'))
        dataCorrect = [correctLabels; correctLabels + data];
    end

    % Drawings
    % bar(labels(2:end), data*100);
    if(is4ViewSupervised)
        maxAngleError = 90;
        angles = labels(labels > 0 & labels <= maxAngleError);
        drawing = data(1:sum(labels > 0 & labels <= maxAngleError))*100;
        if(~isempty(drawing))
            plot([angles; maxAngleError], [drawing; drawing(end)], 'LineSmoothing', 'on', 'LineWidth', 1.1);
        end
        % drawing = zeros(maxAngleError, 1);
        % drawing(round(labels(labels > 0 & labels < maxAngleError))) = data(find(labels > 0 & labels < maxAngleError) - 1)*100;
        % bar(1:1:maxAngleError , drawing);
    else
        maxAngleError = 180;
        angles = labels(labels > 0 & labels <= maxAngleError);
        drawing = data*100;
        angles(drawing == 0) = [];
        drawing(drawing == 0) = [];
        if(~isempty(drawing))
            plot([angles; maxAngleError], [drawing; drawing(end)], 'LineSmoothing', 'on', 'LineWidth', 1.1);
        end
        % drawing = zeros(360, 1);
        % drawing(labels(2:end)) = data*100;
        % bar(1:1:360, drawing);
        
    end
    xlim([0 maxAngleError]);
    ylim([0 100]);
    title(sprintf('Average Angle Error: %.2f° with std %.2fº', angleError, stdDevAngle));
    xlabel('Degrees of angle deviation');
    ylabel('Cumulative % misclassifications');

    % Storage
    strAngle = sprintf('%.2f', angleError);
    strStd = sprintf('%.2f', stdDevAngle);
    pathErrorName = [fileName ' angleError ' strAngle ' ' strStd '.png'];
    if(exist(pathErrorName,'file'))
        pathErrorName = [pathErrorName(1:end-4) ' 2.png'];    
    end
%     saveas(f, pathErrorName);

    drawing = dataCorrect(1:sum(labels <= maxAngleError))*100;
    ax = plot([0; angles], drawing, 'o', 'MarkerSize', 3, 'MarkerFaceColor', [0.4,0.4,0.4], 'Color', [0.5,0.5,0.5], 'LineWidth', 2);
    legend(ax, 'discrete viewpoint deviation','Location','southeast');
    hold on;
    area([0; angles], drawing, 'FaceColor', [0.92,0.92,0.92]);
    
    xlim([0 maxAngleError]);
    ylim([0 100]);
    title(sprintf('Average Angle Error: %.2f° with std %.2fº', angleError, stdDevAngle));
    xlabel('Degrees of angle deviation');
    ylabel('Cumulative % of correct viewpoint classifications');

    % Storage Correct Cumulative Histogram
    pathCorrectName = [fileName ' cumhist_AE ' strAngle ' ' strStd ' (' typeAccuracy ').png'];
    if(exist(pathCorrectName,'file'))
        pathCorrectName = [pathCorrectName(1:end-4) ' 2.png'];    
    end
    saveas(f, pathCorrectName);    

end


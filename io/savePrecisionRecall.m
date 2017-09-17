function [] = savePrecisionRecall(input, AP, AP_VOC, precision, recall, extraText)

    if(nargin < 6)
        extraText = '';
    end
    
    fprintf('Average Precision (AP) with %s using %s training: %2.1f%s\n', input.detTh, input.trainDomain, 100*AP_VOC, extraText);
    prfig = figure('Name', '', 'NumberTitle', 'off');
    set(prfig, 'visible', 'off');
    plot(recall, precision, 'LineWidth', 3);
    strTitle = sprintf('Precision/Recall curve (AP: %2.1f) (AP-VOC: %2.1f)', 100*AP, 100*AP_VOC);
    title(strTitle, 'FontSize', 12, 'FontWeight', 'bold');
    xlabel('Recall'); ylabel('Precision');
    range = [-0.01 1.01]; xlim(range); ylim(range);
    grid on;
    [path, name] = getResultsPath(input);
    saveas(prfig, [ path '/' name '_detAR ' sprintf('%2.1f', 100*AP_VOC) extraText '.png']);

end


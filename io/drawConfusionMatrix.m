function drawConfusionMatrix(numClasses, elemMat, probMat, labels, mFile, mClass, mLabel, typeAccuracy)

    f = figure;
    set(f, 'visible', 'off');
    set(f, 'Position', [500, 100, 800, 800]);
    imagesc(probMat, [-0.02 1]);
    cmap = colormap;
    cmap = [[0.5 0.5 0.5]; cmap];
    colormap(cmap);
    hold on;
    if(strfind(typeAccuracy, 'el'))
        accuracy = sum(diag(elemMat))/sum(sum(elemMat))*100.0;
    elseif(strfind(typeAccuracy, 'pr'))
        emptyElems = sum(diag(probMat) == -1);
        accuracy = (sum(diag(probMat))+emptyElems)/(numClasses-emptyElems)*100.0;
    end
    accuracy = round(accuracy*100.0)/100.0;
    if(strfind(typeAccuracy, 'el'))
        title(sprintf('[%s %s] Accuracy = %.2f%% - %d / %d', ...
            mClass, mLabel, accuracy, sum(diag(elemMat)), sum(sum(elemMat))), 'FontSize', 20);
    else
        title(sprintf('[%s %s] Accuracy = %.2f%%', mClass, mLabel, accuracy), 'FontSize', 20);
    end
    if(numClasses < 20)
        for aind = 1:numClasses
            for bind = 1:numClasses
                if(probMat(aind,bind) > 0.0)
                    if(probMat(aind,bind) >= 0.5)
                        if(strfind(typeAccuracy, 'el'))
                            text(bind,aind,sprintf('%d',elemMat(aind,bind)), 'FontSize', 12, 'Color', 'k', 'HorizontalAlignment', 'center');
                        else
                            text(bind,aind,sprintf('%.2f',probMat(aind,bind)), 'FontSize', 12, 'Color', 'k', 'HorizontalAlignment', 'center');
                        end
                    else
                        if(strfind(typeAccuracy, 'el'))
                            text(bind,aind,sprintf('%d',elemMat(aind,bind)), 'FontSize', 12, 'Color', 'w', 'HorizontalAlignment', 'center');
                        else
                            text(bind,aind,sprintf('%.2f',probMat(aind,bind)), 'FontSize', 12, 'Color', 'w', 'HorizontalAlignment', 'center');
                        end
                    end
                end
            end
        end
        set(gca, 'YTick', 1:numClasses, 'YTickLabel', labels);
        % - No X axis labels: if string too large, bad visualisation
        labels2Char = cellfun(@(x) x(1:2),labels(cellfun('length',labels) > 1),'un',0);
        set(gca, 'XTick', 1:numClasses, 'XTickLabel', labels2Char);
    end
    axis image;
    lab = mClass;
    if(~strcmpi(lab,''))
        lab = [' ' mClass(1:3)];
    end
    % fileName = [mFile ' ' mClass(1:min(3,length(mClass))) '_' mLabel(1:min(2,length(mLabel))) ' ' num2str(accuracy) ' (' typeAccuracy ').png'];
    fileName = [mFile ' ' num2str(accuracy) lab ' (' typeAccuracy ').png'];
    if(exist(fileName,'file'))
        fileName = [fileName(1:end-4) ' 2.png'];    
    end
    saveas(f, fileName);

end


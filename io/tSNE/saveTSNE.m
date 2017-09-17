function saveTSNE(input, path, srcFeatures, srcIds, metadata, tgtFeatures, tgtIds)

    fprintf('tSNE\n');
    for idxAtt = 1:length(metadata)
        att = metadata{idxAtt};
        % Save only with attribute exists and the amount to transfer > 1
        if(isprop(input.sourceDataset, att) && isprop(input.targetDataset, att) && length(input.sourceDataset.(att)) > 1)
            allClasses = unique([input.sourceDataset.(att), input.targetDataset.(att)]);
            srcSamplesClasses = srcIds(:,strcmpi(metadata,att));
            numSrcSamples = zeros(length(srcSamplesClasses),1);
            tgtSamplesClasses = tgtIds(:,strcmpi(metadata,att));
            numTgtSamples = zeros(length(tgtSamplesClasses),1);
            for i = 1:length(allClasses)
                if(isnumeric(allClasses))
                    value = num2str(allClasses(i));
                else
                    value = allClasses{i};
                end
                numSrcSamples(ismember(srcSamplesClasses,value)) = i;
                numTgtSamples(ismember(tgtSamplesClasses,value)) = i;
            end
            computeTSNE(path, att, srcFeatures, numSrcSamples, tgtFeatures, numTgtSamples);
        end
    end

end

function computeTSNE(mDir, name, x1, l1, x2, l2)

    yJoint = tsne([x1;x2], [l1;l2]);
    
    % Define boundaries
%     bMin = min([y1; y2]);
%     bMax = max([y1; y2]);
    bMin = min(yJoint);
    bMax = max(yJoint);
    
    y1 = yJoint(1:size(x1,1),:);
    y2 = yJoint(size(x1,1)+1:end,:);
    
    drawAndSave(mDir, ['/tSNE_S_' name '.png'], y1, l1, bMin, bMax);
    drawAndSave(mDir, ['/tSNE_T_' name '.png'], y2, l2, bMin, bMax);
    drawAndSave(mDir, ['/tSNE_TS_' name '.png'], y1, l1, bMin, bMax, y2, l2);

end

function drawAndSave(mDir, extraPath, y, labels, bMin, bMax, y2, labels2)

    f = figure;
    set(f, 'visible', 'off');
    set(f, 'Position', [500, 100, 800, 800]);

    hold on;
    if (size(y,2) == 1)
        scatter(y, y, 9, labels, 'filled');
    elseif (size(y,2) == 2)
        scatter(y(:,1), y(:,2), 9, labels, 'filled');
    else
        scatter3(y(:,1), y(:,2), y(:,3), 40, labels, 'filled');
    end
    
    if(nargin == 8)
        if (size(y2,2) == 1)
            scatter(y2, y2, 25, labels2, 'x');
        elseif (size(y2,2) == 2)
            scatter(y2(:,1), y2(:,2), 25, labels2, 'x');
        else
            scatter3(y2(:,1), y2(:,2), y2(:,3), 60, labels2, 'x');
        end
    end
    
    axis([bMin(1) bMax(1) bMin(2) bMax(2)])
    saveas(f, [mDir extraPath]);

end
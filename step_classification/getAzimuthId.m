function [ids, views] = getAzimuthId(listAzimuth, azimuth)

    ids = zeros(length(azimuth),1);
    views = zeros(length(azimuth),1);
    [sortedLabels, idxSortedLabels] = sort(listAzimuth, 'descend');
    samples = repmat(azimuth, [1,length(sortedLabels)]);
    uniques = repmat(sortedLabels,[1,size(samples,1)])';
    [valMod, idxMod] = min(abs(samples - uniques),[], 2);
    [val, idx] = min(abs(samples - uniques - 360),[], 2);
    indices = [idx, idxMod];
    [~, idxPickUp] = min([val, valMod],[],2);
    for j = 1:length(listAzimuth)
        for k = 1:length(ids)
            if(indices(k,idxPickUp(k)) == j)
                ids(k) = idxSortedLabels(j);
                views(k) = listAzimuth(ids(k));
            end
        end
    end
            
end
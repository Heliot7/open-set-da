% Convert to RGB for GREYSCALE patches
function img = grey2rgb(img)  
    if(size(img,3) == 1)
        [indPatch, map] = gray2ind(img);
        img = ind2rgb(indPatch, map);
    end
%     if(ismatrix(patch))
%          patch = repmat(patch,[1 1 3]);
%     end
end

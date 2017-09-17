function [scaledImg, scale, pad, scaled_rw] = cropBB(img, row, col, height, width, winSize, border, isScaleInv)

    if(nargin < 8)
        isScaleInv = false;
    end

    winRatio = (winSize(1)-3*border) / (winSize(2)-3*border);
    bbRatio = height / width;
    
    % - Rescale image
    if(~isScaleInv || width > winSize(1) || height > winSize(2))
        if(bbRatio > winRatio)
            scale = (winSize(1)-3*border) / height;
        else
            scale = (winSize(2)-3*border) / width;
        end
        scaledImg = imresize(img, scale, 'bilinear');
    else
        scaledImg = img;
        scale = 1.0;
    end
    
    % "- 1" pixel for hog border gradients
    sHeight = abs(winSize(1) - (round(height*scale) + border*3)) / 2;
    sRow = floor(row*scale - border*1.5 - sHeight);
    
    sWidth = abs(winSize(2) - (round(width*scale) + border*3)) / 2;
    sCol = floor(col*scale - border*1.5 - sWidth);
    
    % Top treatment
    if(sRow < 1)
        minRow = 1;
        topOffset = abs(sRow)+1;
    else
        minRow = sRow;
        topOffset = 0;
    end
    
    % Bottom treatment ("+2pixels border to compensate -1)
    if(sRow+winSize(1)+1 > size(scaledImg,1))
        maxRow = size(scaledImg,1);
        bottomOffset = sRow+winSize(1)+1 - size(scaledImg,1);
    else
        maxRow = sRow+winSize(1)+1;
        bottomOffset = 0;
    end
    
    % Left treatment
    if(sCol < 1)
        minCol = 1;
        leftOffset = abs(sCol)+1;
    else
        minCol = sCol;
        leftOffset = 0;
    end
    
    % Right treatment
    if(sCol+winSize(2)+1 > size(scaledImg,2))
        maxCol = size(scaledImg,2);
        rightOffset = sCol+winSize(2)+1 - size(scaledImg,2);
    else
        maxCol  = sCol+winSize(2)+1;
        rightOffset = 0;
    end
    
    scaledImg = scaledImg(minRow:maxRow,minCol:maxCol, :);

    % - Fill empty holes
    scaledImg = padarray(scaledImg, [topOffset leftOffset], 'replicate', 'pre');
    scaledImg = padarray(scaledImg, [bottomOffset rightOffset], 'replicate', 'post');
    scaledImg = imresize(scaledImg, winSize+2);
    pad = [topOffset, leftOffset, bottomOffset, rightOffset];
    scaled_rw = [max(1,minRow), max(1,minCol)];
    
end

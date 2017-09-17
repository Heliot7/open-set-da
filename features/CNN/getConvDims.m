function [featDim, featSize, imgSize] = getConvDims(descriptor, imgSize)

    global netCaffe;

    if(nargin < 2)
        imgSize = [227 227];
    end
    aux_img = zeros(imgSize(2),imgSize(1), 3);
    % feat = CNN_Caffe('getFeatures', single(aux_img), descriptor(5:end));
    netCaffe.blobs('data').reshape([imgSize(2), imgSize(1), 3, 1]);
    netCaffe.reshape();
    res = netCaffe.forward({aux_img});
    feat = res{1}; % outputs last layer
    featSize = size(feat);
    featDim = prod(featSize);
    
end


function [mhogs] = minimalFeatures(im, signed, num_orient)
    
    %reimplementation from hog4 encoder, unfiltered unnormalized dense
    %orientation histograms; im 3-channel
    
    im=single(im)/255;
    
    [ gx, gy ]          = layer_gradient( im );
    [mag, layer_idx]    = max(gx.^2+gy.^2, [], 3);
    
    gx_max              = layer_pick(gx, layer_idx);
    gy_max              = layer_pick(gy, layer_idx);
    
%     mhogs(:,:,1)=gx_max;
%     mhogs(:,:,2)=gy_max;
%     disp('GRADIENT-ONLY FEATURES')
    
    if signed==1
        exactBin = num_orient .* (1.0 + atan2(gy_max, gx_max)./pi)./2.0001;
    else
        exactBin = num_orient .* (1.0 + atan2(gy_max, gx_max)./pi);
    end

    indLow  = mod(floor(exactBin),num_orient);
    indHigh = mod(ceil(exactBin),num_orient);
    delta   = exactBin - floor(exactBin);
    
    mag_indexLow    = sqrt(1.0-delta).*sqrt( mag );
    mag_indexHigh   = sqrt(delta).*sqrt( mag );
    
    mhogs           = zeros(size(im,1),size(im,2),num_orient);
        
    for i=1:num_orient
        mhogs(:,:,i) = mhogs(:,:,i) + (indLow==(i-1)).*mag_indexLow + (indHigh==(i-1)).*mag_indexHigh;
    end
    
end

function im = layer_pick( im_layered, pick_layer_indx )
    %LAYER_PICK constructs a one layer image by picking values from the
    %given layer index of a multi-layer image
    %
    % Example:
    %
    %     im(:,:,1) = [1 2; 3 4];
    %     im(:,:,2) = [10 20; 30 40];
    %     im(:,:,3) = [100 200; 300 400];
    %
    %     layer = [2 3; 1 2];
    %
    %     layer_pick(im, layer) -> [10 200; 3 40]

    
    [h,w,num_layer] = size(im_layered);
    
    im_perm     = permute(im_layered, [3 1 2]);
    flat_idx    = (1:num_layer:(h*w*num_layer))' + (pick_layer_indx(:)-1); 
    
    im          = reshape(im_perm(flat_idx),[h w]);
end


function [ gx, gy ] = layer_gradient( im )
    %[ gx, gy ] = GRADIENT_LAYERED computes the gradient on every layer of
    %im. The result should correspond to gradient() applied to every layer
    %independently.
    
    
    gx = zeros(size(im), class(im));
    gy = zeros(size(im), class(im));
    
    gx(:,1,:)     = (im(:,3,:)-im(:,1,:))/2;
    gx(:,end,:)   = (im(:,end,:)-im(:,end-2,:))/2;
    
    gy(1,:,:)     = (im(3,:,:)-im(1,:,:))/2;
    gy(end,:,:)   = (im(end,:,:)-im(end-2,:,:))/2;
    
    gx(:,2:(end-1),:) = (im(:,3:end,:)-im(:,1:(end-2),:))/2;
    gy(2:(end-1),:,:) = (im(3:end,:,:)-im(1:(end-2),:,:))/2;
end

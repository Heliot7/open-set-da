classdef Synthetic
    properties
        path = 'Syn\';
        sub_path = 'Output_1d\'; % Output_aerial_high
        % All classes: {'human', 'car', 'bicycle', 'motorbike', 'plane', 'boat'}
%         classes = {'aeroplane', 'bicycle', 'boat', 'bottle', 'bus', 'car', ...
%             'chair', 'diningtable', 'motorbike', 'sofa', 'train', 'tvmonitor'};
        classes = {'car'};
%         classes = {'car','bicycle'};
%         numParts = [8]; % ObjectNet3D (car)
%         numParts = [12]; % ObjectNet3D (car)
%         numParts = [8 11 7 7 12 12 10 12 10 10 17 8]; % ObjectNet3D
        % numParts = [16 11 11 8 8 14 10 8 10 12 7 8]; % Pascal3D
        % Max Viewpoints: 360 (0 = regression, no rounding needed)
        % - Contains elevation and distance
        azimuth = 8;
        parts = []; % Dynamic update
    end
end
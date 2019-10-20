classdef Visda17
    properties
        path = 'Real\DomainAdaptation\Visda17\';
        source = 'train'; % train
        target = 'validation'; % validation
        classes = {'aeroplane', 'bicycle', 'bus', 'car', 'horse', 'knife', ...
        'motorbike', 'person', 'plant', 'skateboard', 'train', 'truck'}; % 6 classes shared with pascal3d
%         classes = {'aeroplane', 'bicycle', 'bus', 'car', 'motorbike', 'train', ... % 6 classes shared with pascal3d
%             'horse', 'knife', 'person', 'plant', 'skateboard', 'truck'}; 
    end
end

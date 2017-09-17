classdef CrossDataset
    properties
        path = 'Real\DomainAdaptation\Tommasi\';
        % For DeCaF7
        % Sparse datasets: 'caltech101', 'caltech256', 'pascal07', 'eth80', 'sun'
        % For AlexNet and VGG (not yet):
        % Sparse datasets: 'AwA'(+), 'PASCAL', 'MSRCORID', 'Caltech256'(+), 'SUN', 'ImageNet'(+), 'MSCOCO'(+)
        % (+) = Very large datasets
        source = 'pascal07';
        target = 'office';
        classes = {};
    end
end

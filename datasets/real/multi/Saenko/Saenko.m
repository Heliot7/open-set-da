classdef Saenko
    properties
        path = 'Real\DomainAdaptation\Saenko\';
        % Included Datasets: AMAZON, DSLR, WEBCAM, CALTECH10
        source = 'AMAZON';
        target = 'DSLR';
        % All classes: {'00 back_pack', '01 bike', '02 calculator', '03 headphones', '04 keyboard',
        % '05 laptop_computer', '06 monitor', '07 mouse', '08 mug', '09 projector'}
        classes = {'00 back_pack', '01 bike', '02 calculator', '03 headphones', '04 keyboard', '05 laptop_computer', ...
            '06 monitor', '07 mouse', '08 mug', '09 projector'};
    end
end

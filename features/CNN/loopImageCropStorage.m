function loopImageCropStorage()

    classes = {'car', 'aeroplane', 'bike', 'boat', 'bus', 'chair', ...
        'diningtable', 'motorbike', 'sofa', 'train', 'tvmonitor'};
    for i = 2:length(classes)
        testCaffeWrapper_train(classes{i});
    end

end


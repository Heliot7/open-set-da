function saveLabelTransfer(input, metaData, transferLabels)

    savePath = [input.PATH_DATA input.targetDataset.path 'transferLabel\'];
    createDir(savePath);
    transferLabel.labels = metaData;
    transferLabel.values = transferLabels;
    
    if(length(input.targetDataset.classes) == 1)
        nameClass = input.targetDataset.classes{1};
    else
        nameClass = 'all';
    end
    
    % Save transfer of labels
    % -> sourceDataset, Feature, isDA, whatDA?
    nameFile = nameTransferLabel(input, nameClass);
    save([savePath nameFile '.mat'], 'transferLabel', '-v7.3');

end


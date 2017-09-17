function nameFile = nameTransferLabel(input, nameClass)

    nameSrc = class(input.sourceDataset);
    if(strcmpi(nameSrc,'Synthetic'))
        nameSrc = [nameSrc '-' input.sourceDataset.sub_path(1:end-1)];
    end
    numViews = length(input.targetDataset.azimuth);
    if(numViews == 1)
        numViews = input.targetDataset.azimuth;
    end
    numViews = num2str(numViews);
    nameFeat = input.typeDescriptor;
    if(~isempty(strfind(nameFeat,'CNN')))
        nameFeat = [input.cnnName '-' input.typeDescriptor];
    end
    if(input.isDA)
        nameDA = '_DA';
    else
        nameDA = '_noDA';
    end
	classifier_name = input.methodSVM;    
    nameFile = [classifier_name '_' nameClass '_' nameSrc '_' numViews '_' nameFeat nameDA];

end


function [path, name] = getResultsPath(input)

    classes = '';
    if(length(input.sourceDataset.classes) >= 5)
        classes = 'MULTI';
    else
        for i = 1:length(input.sourceDataset.classes)
            newClass = input.sourceDataset.classes{i};
            classes = [classes newClass(1:min(4,length(newClass)))];
            if(i ~= length(input.sourceDataset.classes))
                classes = [classes '_'];
            end
        end
    end

    views = '';
    if(strfind(lower(input.typePipeline), 'det'))
        if((strcmpi(input.trainDomain,'src') || strcmpi(input.trainDomain,'both')) && isprop(input.sourceDataset,'azimuth'))
            if(length(input.sourceDataset.azimuth) > 1)
                views = [' az' num2str(length(input.sourceDataset.azimuth))];
            end
        elseif(strfind(input.trainDomain,'tgt') && isprop(input.targetDataset,'azimuth'))
            if(length(input.targetDataset.azimuth) > 1)
                views = [' az' num2str(length(input.targetDataset.azimuth))];                
            end
        end
    elseif(isprop(input.sourceDataset,'azimuth')) % classification
        if(length(input.sourceDataset.azimuth) > 1)
            views = [' az' num2str(length(input.sourceDataset.azimuth))];
        end
    end
    
    extraDA = '';
    if(input.isDA)
        if(strcmpi(input.typeDA,'ATI'))
            extraDA = ['_ATI.' num2str(input.iterResetDA) '.' num2str(input.iterDA)];
            inputLambda = num2str(input.numLambda);
            extraDA = [extraDA 'r' num2str(input.numCorr) 'l' inputLambda '_'];
            if(input.numNN > 0)
                extraDA = [extraDA 'LC' num2str(input.numNN) '_'];
            end
            extraDA = [extraDA num2str(input.numSrcClusters) '_' num2str(input.numTgtClusters)];
            extraDA = [extraDA '_' input.daSpecial];
        elseif(strcmpi(input.typeDA,'gfk'))
            extraDA = ['_gfk' num2str(input.dimPCA)];
        elseif(strcmpi(input.typeDA,'SA'))
            extraDA = ['_SA' num2str(input.dimPCA)];
        elseif(strcmpi(input.typeDA,'TCA'))
            extraDA = ['_TCA' num2str(input.dimPCA)];
        elseif(strcmpi(input.typeDA,'MMDT'))
            extraDA = ['_MMDT' num2str(input.dimPCA)];
        else
            extraDA = ['_' input.typeDA];
        end
    end
       
    srcDataset = class(input.sourceDataset);
    srcDataset = srcDataset(1:min(length(srcDataset),4));
    tgtDataset = class(input.targetDataset);
    tgtDataset = tgtDataset(1:min(length(tgtDataset),4));
    % Add level of fine degree:
    if(strcmpi(class(input.sourceDataset),'Synthetic'))
        srcDataset = [srcDataset input.sourceDataset.sub_path(end-3:end-1)];
    end
    if(isprop(input.sourceDataset,'source'))
        srcDataset = [srcDataset '-' input.sourceDataset.source(1:2)];
        tgtDataset = [tgtDataset '-' input.targetDataset.target(1:2)];
    end
    
    strPose = '';
    if(strfind(lower(input.typePipeline), 'pose'))
        strPose = 'pose';
    end
    strDet = '';
    if(strfind(lower(input.typePipeline), 'det'))
        strDet = 'DET';
        if(input.jointDetPose)
            detType = '_N';
        else
            detType = '_1+N';
        end
        strSS = '';
        if(input.isSelectiveSearch)
            strSS = '(SS)';
        end
        strDet = [strDet input.trainDomain(1) detType strSS '_' mat2str(input.numTrainNeg)];
    end
    strClass = '';
    if(strfind(lower(input.typePipeline), 'class'))
        strClass = 'CL';
        extraSupervised = 'un';
        if(input.isClassSupervised)
            extraSupervised = 'su';
        end
        if(input.is4ViewSupervised)
            extraSupervised = '4v';
        end
        if(input.isOpenset || input.isWSVM)
            extraSupervised = [extraSupervised '_OS'];
        end
        strClass = [strClass '_' extraSupervised];
        if(strcmpi(input.typePipeline, 'CLDET'))
            strClass = [strClass '_'];
        end
    end
    
    descriptor = input.typeDescriptor;
    if(strfind(input.typeDescriptor, 'CNN'))
        posChar = strfind(input.typeDescriptor,'-');
        descriptor = [input.cnnModel '-' input.typeDescriptor(posChar+1:end)];
    end
    
    extraClassifier = '';
    if(strfind(lower(input.typeClassifier), 'svm'))
        extraClassifier = ['-' input.methodSVM]; 
    end
    strAR = '';
    if(~input.keepAR)
        strAR = 'noAR ';
    end
    strZ = '';
    if(input.isZScore)
        strZ = 'z';
    end
    name = [strZ '[' strClass strDet strPose '] ' strAR srcDataset '(' num2str(input.numSrcTrain ) ')' ...
        '-' tgtDataset '(' num2str(input.numTgtTrain) ') ' ...
        classes views ' ' descriptor '_' input.typeClassifier extraClassifier extraDA];    
    path = [input.PATH_RESULTS name];
    
end


function [data, testData] = getEPFL(input, dataset)

    path = [input.PATH_DATA dataset.path];

    % Open information file:
    file = fopen([path 'tripod-seq.txt']);
    
    line = str2num(fgets(file));
    numSeq = line(1);
    numFramesSeq = str2num(fgets(file));
    templateStrFile = strtrim(fgets(file));
    templateStrBB = strtrim(fgets(file));
    list360Pos = str2num(fgets(file));
    listZeroPos = str2num(fgets(file));
    direction = str2num(fgets(file));    
    fclose(file);
    
    % - Get all sequence time steps
    times = cell(numSeq, 1); % cars sequences
    file = fopen([path 'times.txt']);
    for i = 1:numSeq
        times{i} = str2num(fgets(file));
    end
    fclose(file);
    
    % train samples
    data.imgPaths = cell(sum(numFramesSeq(1:10)),1);
    data.annotations.imgId = (1:length(data.imgPaths))';
    data.annotations.classes = repmat(dataset.classes(1),[length(data.imgPaths) 1]);
    data.annotations.BB = zeros(sum(numFramesSeq(1:10)),4);
    data.annotations.vp.azimuth = zeros(sum(numFramesSeq(1:10)),1);
    % test samples
    testData.imgPaths = cell(sum(numFramesSeq(11:20)),1);
    testData.annotations.imgId = (1:length(testData.imgPaths))';
    testData.annotations.classes = repmat(dataset.classes(1),[length(testData.imgPaths) 1]);
    testData.annotations.BB = zeros(sum(numFramesSeq(11:20)),4);
    testData.annotations.vp.azimuth = zeros(sum(numFramesSeq(11:20)),1);
    idx = 1;
    for i = 1:length(numFramesSeq)

        % reset count
        if(i == 11)
            idx = 1;
        end
        
        % - Get annotations
        file = fopen([path sprintf(templateStrBB, i)]);
        fileAnno = textscan(file, '%f %f %f %f');
        fclose(file);
        
        if(i <= 10)
            data.annotations.BB(idx:idx+numFramesSeq(i)-1,:) = [fileAnno{2}, fileAnno{1}, fileAnno{4}, fileAnno{3}];        
        else
            testData.annotations.BB(idx:idx+numFramesSeq(i)-1,:) = [fileAnno{2}, fileAnno{1}, fileAnno{4}, fileAnno{3}];        
        end
        
        timesSeq = times{i};
        step = 360 / timesSeq(list360Pos(i));
        timesSeq = timesSeq - timesSeq(listZeroPos(i));
        for j = 1:numFramesSeq(i)
        
            % - Get path of current image
            if(i <= 10)
                data.imgPaths{idx} = [path sprintf(templateStrFile, i, j)];
            else
                testData.imgPaths{idx} = [path sprintf(templateStrFile, i, j)];
            end
            
            % - Get azimuth data
            angle = mod(direction(i)*timesSeq(j)*step + 360, 360);
            if(i <= 10)
                data.annotations.vp.azimuth(idx) = angle;
            else
                testData.annotations.vp.azimuth(idx) = angle;
            end
            
            idx = idx + 1;
            
        end

    end    
    
    data.annotations.BB = round(data.annotations.BB);
    testData.annotations.BB = round(testData.annotations.BB);
    
end


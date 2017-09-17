function createSentiment()

    path = 'E:\PhD\Data\Real\DomainAdaptation\Sentiment\';
    listDomains = {'books', 'dvd', 'electronics', 'kitchen'};
%     load([path 'allRAW']);
%     raw = cell(length(allRAW),2);
%     for i = 1:length(allRAW)
%         if(mod(i,1000) == 0)
%             fprintf('it: %d/%d\n',i,length(allRAW));
%         end
%         aux_var = allRAW{i};
%         raw(i,1) = aux_var(1);
%         raw(i,2) = aux_var(2);
%     end
%     words = raw(:,1);
%     counter = cellfun(@str2num, raw(:,2));
%     [listWords, ~, ic] = unique(words);
%     wordCounter = zeros(length(listWords), 1);
%     for i = 1:length(listWords)
%         if(mod(i,1000) == 0)
%             fprintf('it: %d/%d\n',i,length(wordCounter));
%         end
%         wordCounter(i) = sum(counter(ic == i)); % 201183
%     end
%     [values, idxSort] = sort(wordCounter,'descend');
%     listWords400 = listWords(idxSort(1:400));
%     save([path 'listWords400'], 'listWords400', '-v7.3');
%     keyboard;
    
    load([path 'listWords400']);
    listDomains = {'books', 'dvd', 'electronics', 'kitchen'};
    for c = 2:4
        raw_pos = getRawData([path listDomains{c} '\positive.review']);
        raw_neg = getRawData([path listDomains{c} '\negative.review']);
        features = zeros(2000,400);
        for i = 1:2000
            fprintf('it: %d\n', i);
            if(i <= 1000)
                review = raw_pos{i};
            else
                review = raw_neg{mod(i,1001)+1};
            end
            for j = 1:length(review)
                word = strsplit(review{j},':');
                isClass = ismember(listWords400, word(1));
                if(sum(isClass) > 0)
                    features(i,isClass) = features(i,isClass) + str2num(word{2});
                end
            end
        end
        save([path listDomains{c} '\' listDomains{c}], 'features', '-v7.3');
%         keyboard;
    end
    keyboard;
    
    idx = 1;
    listWords = cell(500000,1);
    listCount = zeros(500000,1);
    for i = 1:4
        raw = [];
%         raw = [raw; getListRawData([path listDomains{i} '\positive.review'])];
%         raw = [raw; getListRawData([path listDomains{i} '\negative.review'])];
%         save([path 'raw' num2str(i)], 'raw', '-v7.3');
        load([path 'raw' num2str(i)]);
        for j = 1:length(raw)
            fprintf('it: %d\n', j);
            word = raw{j};
            if(isempty(listWords(1:idx-1)) || ~ismember(word(1),listWords(1:idx-1)))
                listWords(idx) = word(1);
                listCount(idx) = str2num(word{2});
                idx = idx + 1;
            else
                isClass = ismember(listWords(1:idx-1),word(1));
                listCount(isClass) = listCount(isClass) + str2num(word{2});
            end
        end
    end
	listWords = listWords(1:idx-1);
    listCount = listCount(1:idx-1);
    [values, idxSort] = sort(listCount,'descend');
    listWords400 = listWords(idxSort(1:400));
    save([path 'listWords'], 'listWords400', '-v7.3');
    return;

end

function raw = getRawData(path)
    
    raw = cell(1000,1);
    file = fopen(path);
    for l = 1:1000
        line = fgetl(file);
        lineSplit = strsplit(line)';
        raw{l} = lineSplit(1:end-1);
    end
	fclose(file);
    
end

function raw = getListRawData(path)
    
    raw = [];
    file = fopen(path);
    for l = 1:1000
        fprintf('line: %d\n', l);
        line = fgetl(file);
        lineSplit = strsplit(line)';
        raw = [raw; cellfun(@strsplit, lineSplit(1:end-1), repmat({':'}, [length(lineSplit)-1 1]), 'UniformOutput', false)];
    end
	fclose(file);
    
end

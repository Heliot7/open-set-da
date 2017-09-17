function updatePrototxt(mDir, type_prototxt, name_prototxt, num_classes, train_params)

    pathPrototxt = [mDir name_prototxt '-' type_prototxt '.prototxt'];
    mDir = strrep(mDir, '\', '/');
    lines = regexp(fileread(pathPrototxt), '\n', 'split');    
    
    name = 0;
    source = 0;
    batch_size = 0;
    lineOutput = 0;
    for i = 1:length(lines)

        % name -> name_prototxt
         if(~isempty(strfind(lines{i},'name:')) && ~name)
             lines{i} = ['name: "' name_prototxt '"'];
             name = name + 1;
         end

        if(strcmpi(type_prototxt,'train'))

            % mean_file -> path MEAN (implicit)
            if(~isempty(strfind(lines{i},'mean_file:')))
                lines{i} = ['mean_file: "' mDir 'imagenet_mean.binaryproto"'];
            % source -> path HDF5 (implicit)
            elseif(~isempty(strfind(lines{i},'source:')) && source == 0) % train
                lines{i} = ['source: "' mDir 'train.txt"'];
                source = source + 1;
            % source -> path HDF5 (implicit)
            elseif(~isempty(strfind(lines{i},'source:')) && source == 1) % val
                lines{i} = ['source: "' mDir 'val.txt"'];
                source = source + 1;
            % batch_size -> train_params.batch_size
            elseif(~isempty(strfind(lines{i},'batch_size:')) && ~batch_size) % train
                lines{i} = ['batch_size: ' num2str(train_params.batch_size_train)];
                batch_size = batch_size + 1;
            % batch_size -> train_params.batch_size
            elseif(~isempty(strfind(lines{i},'batch_size:')) && batch_size == 1) % val
                lines{i} = ['batch_size: ' num2str(round(train_params.batch_size_val))];
                batch_size = batch_size + 1;
            % num_output -> num_classes
            elseif(~isempty(strfind(lines{i},'num_output:'))) % train
                lineOutput = i;
            end

        end
        
    end
    if(lineOutput > 0)
        lines{lineOutput} = ['num_output: ' num2str(num_classes)];
    end
    
    fid = fopen(pathPrototxt, 'w');
    fprintf(fid, '%s\n', lines{:});
    fclose(fid);

end

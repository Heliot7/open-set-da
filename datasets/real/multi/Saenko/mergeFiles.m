function mergeFiles()

    path = 'D:/Core/Caffe-DA/data/office2/';
    formatSpec = '%s\n';
    for i = 1:5
        for s = {'amazon' 'dslr' 'webcam'};
            for t = {'amazon' 'dslr' 'webcam'};
                src = s{1}; tgt = t{1};
                if(strcmpi(src,tgt))
                    continue;
                end
                for ty = {'' 'OS_' 'OSno_'}
                    type = ty{1};
                    list_files = {};
                    % Load src data
                    txt_path_src = [path type src '_source_rnd' mat2str(i) '.txt'];
                    fileID_src = fopen(txt_path_src, 'r');
                    tline = fgetl(fileID_src);
                    while ischar(tline)
                        list_files = [list_files; {tline}];
                        tline = fgetl(fileID_src);
                    end
                    fclose(fileID_src);
                    % Load tgt data
                    txt_path_tgt = [path type tgt '_train_sup_rnd' mat2str(i) '.txt'];
                    fileID_tgt = fopen(txt_path_tgt, 'r');
                    tline = fgetl(fileID_tgt);
                    while ischar(tline)
                        list_files = [list_files; {tline}];
                        tline = fgetl(fileID_tgt);
                    end
                    fclose(fileID_tgt);

                    % Save semi-sup data
                    txt_path = [path type src '_' tgt '_sup_rnd' mat2str(i) '.txt'];
                    fileID = fopen(txt_path, 'w');
                    for idxLines = 1:length(list_files)
                        fprintf(fileID, formatSpec, list_files{idxLines});
                    end
                    fclose(fileID);
                end
            end
        end
    end
   

end


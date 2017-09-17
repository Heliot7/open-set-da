function data = setupData(dataset_info, data)

    if(isprop(dataset_info,'classes'))
        data.classSetup = repmat(struct('className',''), length(dataset_info.classes), 1);
        for c = 1:length(dataset_info.classes)
            data.classSetup(c).className = dataset_info.classes{c};            
            % Estimate the aspect ratio for all class samples (+ per view)
            % NOW COMMENT - NEED TO HANDLE CASE WITH NO INFO (real data i.e.)
            if(isfield(data.annotations,'BB'))
                isClass = ismember(data.annotations.classes, dataset_info.classes{c});
                BBs = data.annotations.BB(isClass,:);
                data.classSetup(c).muSize = mean(double(BBs(:,3:4)));
                data.classSetup(c).stdSize = std(double(BBs(:,3:4)));
                data.classSetup(c).ar = data.classSetup(c).muSize(1) / data.classSetup(c).muSize(2);
                % Comment: Add new metadata attributes if necessary
                % - Currently: viewpoint
                [~, ~, metadata] = getIdLabels(dataset_info, data.annotations);
                metadata = metadata(~strcmpi(metadata,'classes'));
                [idLabelsStr, nameLabels] = getIdLabels(dataset_info, data.annotations, metadata);
                if(~isempty(nameLabels))
                    data.classSetup(c).metadata = repmat(struct('ar',1), size(nameLabels,1), 1);
                    for i = 1:size(nameLabels,1)
                        isClass = ismember(idLabelsStr, nameLabels{i,:}, 'rows');
                        viewBBs = data.annotations.BB(isClass,:);
                        if(sum(isClass) > 1)
                            data.classSetup(c).metadata(i).muSize = mean(double(viewBBs(:,3:4)));
                            data.classSetup(c).metadata(i).stdSize = std(double(viewBBs(:,3:4)));
                            data.classSetup(c).metadata(i).ar = data.classSetup(c).metadata(i).muSize(2) / data.classSetup(c).metadata(i).muSize(2);
                        elseif(sum(isClass) == 1)
                            data.classSetup(c).metadata(i).muSize = double(viewBBs(3:4));
                            data.classSetup(c).metadata(i).stdSize = 0.0;
                            data.classSetup(c).metadata(i).ar = viewBBs(3) / viewBBs(4);
                        else % none
                            data.classSetup(c).metadata(i).muSize = 0.0;
                            data.classSetup(c).metadata(i).stdSize = 1.0;
                            data.classSetup(c).metadata(i).ar = 1.0;
                        end
                    end
                end
            end
        end
    end
    
end

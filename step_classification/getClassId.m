function ids = getClassId(listClasses, classes)

    % cellfind = @(string)(@(cell_contents)(strcmp(string,cell_contents)));
    ids = zeros(length(classes),1);
    for idxClass = 1:length(listClasses)
        % isClass = cellfun(cellfind(listClasses(idxClass)),classes,'UniformOutput',false);
        % ids(logical(cellfun(@sum,isClass))) = idxClass;
        isClass = sum(ismember(classes,listClasses(idxClass)),2);
        % ids(isClass == size(listClasses,2)) = idxClass;
        ids(logical(isClass)) = idxClass;
    end
            
end
% VARARGIN: Size must be pair: tagMetadata, value
function ids = retrieveId(metadata, labels, varargin)

    isMetadata = zeros(length(labels), length(varargin)/2);
    cellfind = @(string)(@(cell_contents)(strcmp(string,cell_contents)));
    for i = 1:2:length(varargin)
        isIncluded = strcmpi(metadata, varargin{i});
        label = labels(:,isIncluded);
        value = varargin{i+1};
        for v = 1:length(value)
            isClass = cellfun(cellfind(value{v}),label,'UniformOutput',false);
            isMetadata(logical(cellfun(@sum,isClass)),(i+1)/2) = 1;
        end
    end
    ids = find(prod(isMetadata,2));
    ids = labels(ids,:);
    
end
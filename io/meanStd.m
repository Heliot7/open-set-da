function [m, s] = meanStd(values)

    if(nargin == 0)
        values = [];
    end

    m = -1;
    s = -1;
    if(~isempty(values))
        m = mean(values);
        s = std(values);
    end
    
end



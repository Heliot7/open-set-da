function listAzimuths = getDiscreteAzimuths(numViews, is4ViewSupervised)
    
    stepSize = 360.0/numViews;
    listAzimuths = zeros(numViews,1);
    
    if(is4ViewSupervised) 
        % Fine views should be multiple of 4
        if(mod(numViews,4) ~= 0)
            error('WARNING: %d views is not a multiple of 4\n', numViews);
        end
        % Start from front view left side (-45 degrees)
        % offsetInit = 315;
        % Start from the front view (0 degrees)
        offsetInit = -stepSize/2.0;
    else
        offsetInit = -stepSize/2.0;
    end
    listAzimuths(1) = offsetInit + stepSize/2.0;
    for i = 2:numViews
        listAzimuths(i) = mod(listAzimuths(i-1) + stepSize + 360, 360);
    end
    
end

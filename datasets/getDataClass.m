function data = getDataClass(input, data, strObj)

    isClass = ismember(data.annotations.classes, strObj);
    data.annotations.classes = data.annotations.classes(isClass);
    if(isfield(data.annotations, 'imgId'))
        data.annotations.imgId = data.annotations.imgId(isClass);
    end
    if(isfield(data.annotations, 'BB'))
        data.annotations.BB = data.annotations.BB(isClass,:);
    end
    if(isfield(data.annotations, 'parts'))
        data.annotations.parts = data.annotations.parts(isClass,:);
    end
    if(isfield(data.annotations, 'vp'))
        if(isfield(data.annotations.vp, 'azimuth'))
            data.annotations.vp.azimuth = data.annotations.vp.azimuth(isClass);
        end
        if(isfield(data.annotations.vp, 'elevation'))
            data.annotations.vp.elevation = data.annotations.vp.elevation(isClass);
        end
        if(isfield(data.annotations.vp, 'distance'))
            data.annotations.vp.distance = data.annotations.vp.distance(isClass);
        end
        if(isfield(data.annotations.vp, 'plane'))
            data.annotations.vp.plane = data.annotations.vp.plane(isClass);
        end
    end
    if(isfield(data.annotations, 'camera'))
        if(isfield(data.annotations.camera, 'focal'))
            data.annotations.camera.focal = data.annotations.camera.focal(isClass);
        end
        if(isfield(data.annotations.camera, 'px'))
            data.annotations.camera.px = data.annotations.camera.px(isClass);
        end
        if(isfield(data.annotations.camera, 'py'))
            data.annotations.camera.py = data.annotations.camera.py(isClass);
        end
        if(isfield(data.annotations.camera, 'viewport'))
            data.annotations.camera.viewport = data.annotations.camera.viewport(isClass);
        end
    end
    if(isfield(data, 'partLabels'))
        data.partLabels = data.partLabels(ismember(input.sourceDataset.classes, strObj));
    end

end


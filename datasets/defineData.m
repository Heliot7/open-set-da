function data = defineData()

    data.imgPaths = []; data.annotations.imgId = []; data.annotations.BB = [];
    data.annotations.classes = []; data.annotations.parts = [];
    data.annotations.vp.azimuth = []; data.annotations.vp.elevation = [];
    data.annotations.vp.distance = []; data.annotations.vp.plane = [];
    data.annotations.camera.px = []; data.annotations.camera.py = [];
    data.annotations.camera.focal = []; data.annotations.camera.viewport = []; 
    data.partLabels = [];

end
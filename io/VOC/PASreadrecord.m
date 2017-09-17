function rec = PASreadrecord(path)

if length(path)<4
    error('unable to determine format: %s',path);
end

if strcmp(path(end-3:end),'.txt')
    rec=PASreadrectxt(path);
elseif strcmpi(path(end-3:end),'.xml')
    rec=VOCreadrecxml(path);
elseif strcmpi(path(end-3:end),'.mat')
    rec_pre=load(path);
    rec.filename = rec_pre.record.filename;
    rec.objects = rec_pre.record.objects;
    rec.size = rec_pre.record.size;
    rec.database = rec_pre.record.database;
    rec.imgsize = rec_pre.record.imgsize;
end

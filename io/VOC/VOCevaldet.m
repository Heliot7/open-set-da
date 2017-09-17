function [rec,prec,ap] = VOCevaldet(VOCopts,id,cls,draw)

% load test set

cp=sprintf(VOCopts.annocachepath,VOCopts.testset);
if exist(cp,'file')
    fprintf('%s: pr: loading ground truth\n',cls);
    load(cp,'gtids','recs','recs_i3d','maxNumIds');
else
    [gtids, t] = textread(sprintf(VOCopts.imgsetpath,VOCopts.testset),'%s %d');
    maxNumIds = length(gtids);
    if(isfield(VOCopts, 'imgsetpath_i3d'))
        [gtidsI3D, tI3D] = textread(sprintf(VOCopts.imgsetpath_i3d, cls, VOCopts.testset),'%s %d');
        gtids = [gtids; gtidsI3D];
        t = [t; tI3D];
    end
    for i=1:length(gtids)
        % display progress
        if toc>1
            fprintf('%s: pr: load: %d/%d\n',cls,i,length(gtids));
            drawnow;
            tic;
        end

        % read annotation
        if(i <= maxNumIds)
            recs(i) = PASreadrecord(sprintf(VOCopts.annopath, gtids{i}));
        else % I3D extra
            recs_i3d(i) = PASreadrecord(sprintf(VOCopts.annopath_i3d, cls, gtids{i}));
        end
    end
    if(~isfield(VOCopts, 'imgsetpath_i3d'))
        save(cp,'gtids','recs','maxNumIds','-v7.3');
    else
        save(cp,'gtids','recs','recs_i3d','maxNumIds','-v7.3');
    end
end

fprintf('%s: pr: evaluating detections\n',cls);

% hash image ids
% hash=VOChash_init(gtids);
        
% extract ground truth objects

npos=0;
gt(length(gtids))=struct('BB',[],'diff',[],'det',[]);
for i=1:length(gtids)
    % extract objects of class
    if(i <= maxNumIds)
        r = recs(i);
    else
        r = recs_i3d(i);
    end
    clsinds=strmatch(cls,{r.objects(:).class},'exact');
    gt(i).BB=cat(1,r.objects(clsinds).bbox)';
    gt(i).diff=[r.objects(clsinds).difficult];
    gt(i).det=false(length(clsinds),1);
    npos=npos+sum(~gt(i).diff);
end

% load results
[ids,confidence,b1,b2,b3,b4]=textread(sprintf(VOCopts.detrespath,id,cls),'%s %f %f %f %f %f');
BB=[b1 b2 b3 b4]';

% sort detections by decreasing confidence
[sc,si]=sort(-confidence);
ids=ids(si);
BB=BB(:,si);

% assign detections to ground truth objects
nd=length(confidence);
tp=zeros(nd,1);
fp=zeros(nd,1);
tic;
out_imgs = false(nd,1);
figure;
for d = 1:nd
    
    % display progress
    if toc>1
        fprintf('%s: pr: compute: %d/%d\n',cls,d,nd);
        drawnow;
        tic;
    end
    
    % find ground truth image % remove full path
    i = find(ismember(gtids,ids{d})); % VOChash_lookup(hash,ids{d});
    if isempty(i)
        if(si(d) >= maxNumIds)
            out_imgs(d) = true;
        end
        show = false;
        continue; % no matching, continue and remove later (on I3D)
        % error('unrecognized image "%s"',ids{d});
    elseif length(i)>1
        error('multiple image "%s"',ids{d});
    else
        img_name = ids{d};
        if(strcmpi(img_name(1),'n'))
            imshow(imread(['Z:\PhD\Data\Real\Multi\ImageNet3D\Images\car_imagenet\' img_name '.jpeg']));
        else
            imshow(imread(['Z:\PhD\Data\Real\Multi\PASCAL_VOC12\train\JPEGImages\' img_name '.jpg' ]));
        end
        hold on;
        bb = BB(:,d);
        listXYWH = [bb(1) bb(2) bb(3)-bb(1) bb(4)-bb(2)];
        rectangle('position', listXYWH, 'LineWidth', 1, 'EdgeColor', [0 1 0]);
        show = true;
    end

    % assign detection to ground truth object if any
    bb = BB(:,d);
    ovmax=-inf;
    for j=1:size(gt(i).BB,2)
        bbgt=gt(i).BB(:,j);
        if(show)
            listXYWH = [bbgt(1) bbgt(2) bbgt(3)-bbgt(1) bbgt(4)-bbgt(2)];
            rectangle('position', listXYWH, 'LineWidth', 1, 'EdgeColor', [0.3 0.3 1.0])
        end
        if(show && j == size(gt(i).BB,2))
            hold off;
        end
        bi=[max(bb(1),bbgt(1)) ; max(bb(2),bbgt(2)) ; min(bb(3),bbgt(3)) ; min(bb(4),bbgt(4))];
        iw=bi(3)-bi(1)+1;
        ih=bi(4)-bi(2)+1;
        if iw>0 & ih>0
            % compute overlap as area of intersection / area of union
            ua=(bb(3)-bb(1)+1)*(bb(4)-bb(2)+1)+...
               (bbgt(3)-bbgt(1)+1)*(bbgt(4)-bbgt(2)+1)-...
               iw*ih;
            ov=iw*ih/ua;
            if ov>ovmax
                ovmax=ov;
                jmax=j;
            end
        end
    end
    % assign detection as true positive/don't care/false positive
    if ovmax>=VOCopts.minoverlap
        if ~gt(i).diff(jmax)
            if ~gt(i).det(jmax)
                tp(d)=1;            % true positive
		gt(i).det(jmax)=true;
            else
                fp(d)=1;            % false positive (multiple detection)
            end
        end
    else
        fp(d)=1;                    % false positive
    end
end

if(length(ids) > maxNumIds)
    fp(out_imgs) = [];
    tp(out_imgs) = [];
end

% compute precision/recall
fp=cumsum(fp);
tp=cumsum(tp);
rec=tp/npos;
prec=tp./(fp+tp);

ap=VOCap(rec,prec);

if draw
    % plot precision/recall
    plot(rec,prec,'-');
    grid;
    xlabel 'recall'
    ylabel 'precision'
    title(sprintf('class: %s, subset: %s, AP = %.3f',cls,VOCopts.testset,ap));
end

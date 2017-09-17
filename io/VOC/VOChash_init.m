function hash = VOChash_init(strs)

% hsize=4999;
hsize=numel(strs)*2;
hash.key=cell(hsize,1);
hash.val=cell(hsize,1);

for i=1:numel(strs)
    s=strs{i};
    substr = s([3:4 6:11 13:end]);
    substr(strfind(substr,'_')) = [];
    h=mod(str2double(substr),hsize)+1;
    j=numel(hash.key{h})+1;
    hash.key{h}{j}=strs{i};
    hash.val{h}(j)=i;
end


function ind = VOChash_lookup(hash,s)

hsize=numel(hash.key);
substr = s([3:4 6:11 13:end]);
substr(strfind(substr,'_')) = [];
h=mod(str2double(substr),hsize)+1;
ind=hash.val{h}(strmatch(s,hash.key{h},'exact'));

function createDir(mDir)

    if(~exist(mDir, 'dir'))
        mkdir(mDir);
    end

end
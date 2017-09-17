function removeDir(mDir)

    if(exist(mDir, 'dir'))
        rmdir(mDir, 's');
    end

end
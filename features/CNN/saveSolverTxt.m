function saveSolverTxt(folderPath, NAME_NETWORK, solver_params)

    fileID = fopen([folderPath NAME_NETWORK '-solver.prototxt'],'w');
    if(fileID < 0)
        error('[[Caught ERROR: solver file could not be opened!');
    end
    
    fprintf(fileID, 'net: "%s"\n', solver_params.solverPath);
    % fprintf(fileID, 'type: "%s"\n', solver_params.solverType);
    fprintf(fileID, 'test_iter: %s\n', num2str(solver_params.num_iter));
    fprintf(fileID, 'test_interval: %s\n', num2str(solver_params.test_interval));
    fprintf(fileID, 'base_lr: %s\n', num2str(solver_params.base_lr));
    fprintf(fileID, 'lr_policy: "%s"\n', solver_params.lr_policy);
    fprintf(fileID, 'gamma: %s\n', num2str(solver_params.gamma));
    fprintf(fileID, 'momentum: %s\n', num2str(solver_params.momentum));
    % fprintf(fileID, 'momentum2: %s\n', num2str(solver_params.momentum2));
    % fprintf(fileID, 'delta: %s\n', num2str(solver_params.delta));
    fprintf(fileID, 'stepsize: %s\n', num2str(solver_params.stepsize));
    fprintf(fileID, 'weight_decay: %s\n', num2str(solver_params.weight_decay));
    fprintf(fileID, 'display: %s\n', num2str(solver_params.display));
    fprintf(fileID, 'max_iter: %s\n', num2str(solver_params.max_iter));
    fprintf(fileID, 'snapshot: %s\n', num2str(solver_params.snapshot));
    fprintf(fileID, 'snapshot_prefix: "%s"\n', solver_params.snapshot_prefix);
    fprintf(fileID, 'solver_mode: %s\n', solver_params.solver_mode);
    
    fclose(fileID);
end
function cnnSetupTraining(source_cnn, name_cnn, data_format, data_path, num_classes)

    % Example:
    % cnnSetupTraining('AlexNet','lmdb', % [input.PATH_DATA input.sourceDataset.path 'img_samples\'], 2)
    input = InputParamters;
    % (1) Create new CNN model folder
    createDir([input.PATH_CNN name_cnn]);
    % (2) Set up solver
    % - Update solver
    solver_params = defineSolver(savePath, NAME_NETWORK);
    saveSolverTxt(savePath, NAME_NETWORK, solver_params);
    % - Update prototxt of models with proper path of image lists    
    train_params = defineTraining();
    % > train_model
    type_file = 'train';
    updatePrototxt(savePath, type_file, NAME_NETWORK, numClasses, train_params);
    % > deploy_mode
    type_file = 'deploy';
    updatePrototxt(savePath, type_file, NAME_NETWORK, numClasses, train_params);

end

function solver_params = defineSolver(folderPath, NAME_NETWORK)

    folderPath = strrep(folderPath, '\', '/');

    % Path of training file
    solver_params.solverPath = [folderPath NAME_NETWORK '-train.prototxt'];
    % Type of optimiser
    solver_params.solverType = 'Adam';
    % Number of test iterations (num test images / batch size)
    solver_params.num_iter = 1000;
    % Every how many iterations testing is performed
    solver_params.test_interval = 1000;
    % The base learning rate, momentum and the weight decay of the network.
    % The learning rate policy
    solver_params.lr_policy = 'step';
    solver_params.base_lr = 0.001; % 0.01 new 0.001 ft
    solver_params.momentum = 0.9;
    % momentum2 and delta for Adam optimiser
    solver_params.momentum2 = 0.999;
    solver_params.delta = 0.0000001;
    solver_params.weight_decay = 0.0005;
    solver_params.gamma = 0.1;
    % Drop of the learning rate everz X iterations
    solver_params.stepsize = 2500;
    % Number of iteration for display
    solver_params.display = 25;
    % Maximum number of iterations
    solver_params.max_iter = 10000;
    % Snapshot: intermediate results
    solver_params.snapshot = 2500;
    solver_params.snapshot_prefix = [folderPath NAME_NETWORK];
    % Solver mode: CPU or GPU
    solver_params.solver_mode = 'GPU';

end

function train_params = defineTraining()

    train_params.batch_size_train  = 256;
    train_params.batch_size_val  = 50;

end
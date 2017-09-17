function cnnStoreImageSamples(typeStorage, dataset, classes, sampleSize, numNegSamples, varargin)

    input = InputParameters;
    % 'lmdb', 'imgData', 'hdf5'
    if(~exist('typeStorage','var'))
        typeStorage = 'lmdb';
    end
    input.sourceDataset = feval(dataset);
    if(~exist('classes','var'))
        classes = 'car';
    end
    input.sourceDataset.classes = classes;
    if(~exist('sampleSize','var'))
        sampleSize = [256, 256];
    end
    if(~exist('numNegSamples','var'))
        numNegSamples = 10000;
    end
    % [[ varargin{1} = azimuth, varargin{2} = sub-classes ]    
    if(isprop(input.sourceDataset, 'azimuth'))
        try
            input.sourceDataset.azimuth = varargin{1};
        catch
            fprintf('WARNING: No viewpoint param assigned!\n');
        end
    end
    if(isprop(input.sourceDataset, 'source'))
        try
            input.sourceDataset.source = varargin{2};
        catch
            fprintf('WARNING: No source-class param assigned!\n');
        end
    end
    
    % - Parameters
    input.keepAR = true;
    input.isScaleInvariance = true;

    % Store Data
    switch(typeStorage)
        case 'hdf5'
            saveDataToHDF5(input, sampleSize, numNegSamples);    
        case {'lmdb', 'imgData'}
            saveDataToImg(input, sampleSize, numNegSamples);
            if(strcmpi('lmdb',typeStorage))
                savePath = [input.PATH_DATA input.sourceDataset.path 'img_samples\'];
                LMDB_path = 'D:\Core\Caffe-git\VS2015\bin\Release\';
                commandCreateLMDB = [LMDB_path 'convert_imageset --shuffle ' savePath ' ' savePath 'train.txt ' savePath];
                system([commandCreateLMDB 'train_lmdb']);
                commandCreateLMDB = [LMDB_path 'convert_imageset --shuffle ' savePath ' ' savePath 'val.txt ' savePath];
                system([commandCreateLMDB 'val_lmdb']);
            end
    end

end
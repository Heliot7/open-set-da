function classifiers = trainLabels(input, data, features, idData, strLabels, typeSVM, isStorage, isVerbose)

    if(nargin < 8)
        isVerbose = true;
    end
    if(nargin < 7)
        isStorage = false;
    end
    if(nargin < 6)
        typeSVM = input.typeClassifier;
    end
    
	% Multi-class classification:
    numClasses = size(strLabels,1);
    if(strcmpi(input.methodSVM,'liblinear'))
        numClassifiers = numClasses;
        if(numClassifiers == 2)
            numClassifiers = 1;
        end
    elseif(~isempty(strfind(input.methodSVM,'libsvm')))
        numClassifiers = numClasses*(numClasses-1)/2;
        if(strcmpi(input.methodSVM,'libsvm-open'))
            numClassifiers = numClasses;
        end
    else
        error('[[Caught ERROR: wrong multi-classification method]]');
    end
    w_pos = zeros(numClassifiers, size(features,2));
    w_neg = zeros(numClassifiers, size(features,2));
    bias = -Inf*ones(numClassifiers,1);
        
	if(isVerbose)
		fprintf('Training of multi-class classification with %s... ', input.methodSVM);
	end
	t = tic;
	ids = zeros(size(features,1),1);
	for i = 1:length(strLabels)
		ids(prod(ismember(idData,strLabels(i,:)),2,'native')) = i;
	end
	
	% Libsvm (always OVO) - Openset Variation
	if(strcmpi(input.methodSVM,'libsvm-open'))
		% [model_0, model_open_0] = svmtrain_open(ids, sparse(double(features)), '-t 0 -s 0 -q -c 0.01');
		[model, model_open] = svmtrain_open(ids, sparse(double(features)), '-s 7 -t 2 -q -c 0.01 -G 5 10 ');
        % [model, model_open] = svmtrain_open(ids, sparse(double(features)), '-s 10 -t 2 -q -c 0.01');
		% model_zero = svmtrain(ids, sparse(double(features)), '-t 0 -q -c 0.01');
		classifiers.model = model;
		classifiers.model_open = model_open;
		
	end
	% Libsvm (always OVO)
	if(strcmpi(input.methodSVM,'libsvm'))

		t_param = 0;
        if(strcmpi(typeSVM,'SVM'))
            t_param = 1; % polynomial
        end
        if(input.CV_LSVM)
			C = [0.001 0.01 0.1 1.0 10 100 1000 10000];   
			model = zeros(size(C,2),1);
			parfor i = 1:size(C,2)
				model(i) = svmtrain(ids, sparse(double(features)), ['-t ' num2str(t_param) ' -c ' num2str(C(i)) ' -v 2 -q']);
			end	
			[~, indx] = max(model); 
			input.C_LSVM = C(indx);
        end
		model = svmtrain(ids, sparse(double(features)), sprintf('-t %d -q -c %d', t_param, input.C_LSVM));
		classifiers.model = model;
		
	% Liblinear (always OVA)
	elseif(strcmpi(input.methodSVM,'liblinear'))

		if(input.CV_LSVM)
			bestC = trainLibLinear(ids, sparse(double(features)), '-s 2 -C -v 2 -B 1 -q');
			input.C_LSVM = bestC;
		end
		model = trainLibLinear(ids, sparse(double(features)), sprintf('-s 2 -n 12 -B 1 -c %d -q', input.C_LSVM));
		bias = model.w(:,end); % [w; bias] (when -B 1)
		w_pos = model.w(:,1:end-1);
		if(numClassifiers == 2)
			w_pos = [w_pos; -1*w_pos];
			bias = [bias; -bias];
		end
		classifiers.model = model;

	end
	if(isVerbose)
		fprintf('in %.2f sec\n',toc(t));
	end
    
    classifiers.w_pos = w_pos;
    classifiers.w_neg = w_neg;
    classifiers.bias = bias;

end

function [input, srcData, srcFeatures, tgtData, tgtFeatures, testData, testFeatures] = intersectionCrossDataset(input, srcData, srcFeatures, tgtData, tgtFeatures)

    testData = [];    
    testFeatures = [];

    srcClasses = input.sourceDataset.classes;
    tgtClasses = input.targetDataset.classes;
    
    jointClasses = srcClasses(ismember(srcClasses,tgtClasses));
    if(length(jointClasses) < length(srcClasses))
        newClasses = sort_nat([jointClasses {'zz_unknown'}]);
        % Source domain
        bgClasses = ~ismember(srcData.annotations.classes, jointClasses);
        % Taking also unknown
        input.sourceDataset.classes = newClasses;
        srcData.annotations.classes(bgClasses) = repmat({'zz_unknown'}, [sum(bgClasses) 1]);
        % Using only the shared classes
    %     input.sourceDataset.classes = jointClasses;
    %     srcData.annotations.classes(bgClasses) = [];
    %     srcFeatures(bgClasses,:) = [];

        % Target domain
        input.targetDataset.classes = newClasses;
        bgClasses = ~ismember(tgtData.annotations.classes, jointClasses);
        tgtData.annotations.classes(bgClasses) = repmat({'zz_unknown'}, [sum(bgClasses) 1]);
        
%         unknownSamples = sum(ismember(tgtData.annotations.classes,'unknown'));
%         totalSamples = size(tgtFeatures,1);
%         fprintf('Ratio for %s-%s: %.2f \n', input.sourceDataset.source, input.sourceDataset.target, unknownSamples / totalSamples);
%         keyboard;
        
        if(input.isClassSupervised)
            testData.annotations.classes = [];
            newTgtFeatures = [];
            newTgtLabels = [];
            if(strcmpi(input.typeWildSupervision,'Office'))
                for i = 1:length(newClasses)
                    takeSamples = find(ismember(tgtData.annotations.classes,newClasses{i}));
                    if(~strcmpi(newClasses{i},'zz_unknown'));
                        % 3 samples in training target
                        newTgtFeatures = [newTgtFeatures; tgtFeatures(takeSamples(1:3),:)];
                        newTgtLabels = [newTgtLabels; repmat(newClasses(i),[3,1])];
                        % the rest as test target
                        testFeatures = [testFeatures; tgtFeatures(takeSamples(4:end),:)];
                        testData.annotations.classes = [testData.annotations.classes; repmat(newClasses(i),[length(takeSamples)-3,1])];
                    else
%                         % 3 samples in training target
%                         newTgtFeatures = [newTgtFeatures; tgtFeatures(takeSamples(1:25),:)];
%                         newTgtLabels = [newTgtLabels; repmat(newClasses(i),[25,1])];
%                         % the rest as test target
%                         testFeatures = [testFeatures; tgtFeatures(takeSamples(26:end),:)];
%                         testData.annotations.classes = [testData.annotations.classes; repmat(newClasses(i),[length(takeSamples)-25,1])];
                        testFeatures = [testFeatures; tgtFeatures(takeSamples,:)];
                        testData.annotations.classes = [testData.annotations.classes; repmat(newClasses(i),[length(takeSamples),1])];
                    end
                end
            else % it is a number (how many we take)
                numSemi = str2double(input.typeWildSupervision);
                aux_features = tgtFeatures(1:numSemi,:);
                aux_labels= tgtData.annotations.classes(1:numSemi);
                
                % Sort labels:
                newTgtLabels = [];
                newTgtFeatures = [];
                for i = 1:length(newClasses)
                    newTgtLabels = [newTgtLabels; aux_labels(ismember(aux_labels,newClasses{i}))];
                    newTgtFeatures = [newTgtFeatures; aux_features(ismember(aux_labels,newClasses{i}),:)];
                end
                testFeatures = tgtFeatures(numSemi+1:end,:);
                testData.annotations.classes = tgtData.annotations.classes(numSemi+1:end);
            end
            tgtFeatures = newTgtFeatures;
            tgtData.annotations.classes = newTgtLabels;
        end
    end

end

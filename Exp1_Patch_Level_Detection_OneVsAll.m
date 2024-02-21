clear all
close all
clc
% delete(findall(0));
% close all training plots
% delete(findall(0));
% --------------------------------------------------------------------------
% Step 1: Basic Settings
% --------------------------------------------------------------------------
dataset_partition = ["Test", "Val1", "Val2", "Val3", "Val4"]
%                 k =  1       2        3       4      5
k = 1;
class_names = ["NC", "G3", "G4", "G5"]
class = 4; 
% --------------------------------------------------------------------------
% Step 2: Load Labels for the selected Dataset Partition
% --------------------------------------------------------------------------
dataSetDir = 'D:\Rnd\Frameworks\Datasets\SICAPv2\';
[labels_ALL] = readSICAPV2dataset(dataSetDir,1);
% return

doTraining = true;
 doTraining = false;

for experiment=1:1
    for k=1:1
        for class=2:4
            fn = fieldnames(labels_ALL);
            testData = labels_ALL.(fn{(k-1)*2+1});
            if(doTraining)
                trainData = labels_ALL.(fn{(k-1)*2+2});
                train_one_vs_all_models(trainData,testData,k,dataset_partition,class,class_names,experiment);
            end
        end
        evaluate_detector(k, experiment, class_names, dataset_partition, testData);
    end
end

return;



function train_one_vs_all_models(trainData,testData,k,dataset_partition,class,class_names,experiment)
mdlName = ['GG_Classification_',char(class_names(class)),'_',char(dataset_partition(k)),'_ResNet18_Exp',num2str(experiment,'%.2d')]
% mdlName = ['GG_Classification_',char(class_names(class)),'_',char(dataset_partition(k)),'_InceptionV3_Exp',num2str(experiment,'%.2d')]
patch_size = [224 224];
miniBatchSize = 128;
% --------------------------------------------------------------------------
% Step 3: Prepare image datastore
% --------------------------------------------------------------------------
[train_imds,test_imds,train_imds_aug,test_imds_aug,numTrainImages,numTestImages] = prepare_imds_training(trainData,testData,patch_size,miniBatchSize,class);
classNames = unique(train_imds.Labels)
N_examples = countEachLabel(train_imds)
N_examples = table2array(N_examples(:,2));
classWeights = (sum(N_examples)./N_examples)'
% normalize weights
% classWeights = classWeights/sum(classWeights);
% return
% keyboard
classWeights = ones(1,2);

%   classWeights =classWeights/min(classWeights);
%   classWeights(2) = classWeights(2)*0.5;
%    classWeights(1) = 0.5;
%   classWeights = classWeights*0.25;

 classWeights
 
% classWeights = [1 10];
% preview(train_imds_aug)
% preview(train_imds_aug)
% return
% --------------------------------------------------------------------------
% Step 4: Build Model LGraph for training
% --------------------------------------------------------------------------
load_weights = 1;
output_type = 1;%0-Regression, 1-Classification, 2- Multi-label Classification
output_size = 2;
lgraph = customize_ResNet18(patch_size,output_type,output_size,load_weights,classNames,classWeights);
% lgraph = customize_inceptionV3(patch_size,output_type,output_size,load_weights,classNames,classWeights);
%lgraph = SetLearningRateWeights(lgraph);
% analyzeNetwork(lgraph)
% return

% --------------------------------------------------------------------------
% Step 5: Train the model
% --------------------------------------------------------------------------
maxEpochs = 5;
initialLearningRate = 1e-3;%2e-3;
% Plots="training-progress", ...
    
options = trainingOptions("adam", ...
    Plots="training-progress", ...
    MaxEpochs=maxEpochs, ...
    InitialLearnRate=initialLearningRate, ...
    LearnRateSchedule="piecewise", ...
    LearnRateDropPeriod=5, ...%round(maxEpochs/4), ...
    LearnRateDropFactor=0.1, ...    
    ValidationData=test_imds_aug, ...
    ValidationFrequency=round(2*numTrainImages/miniBatchSize), ... %every two epoch
    Verbose=true, ...
    ExecutionEnvironment="auto", ...
    DispatchInBackground=true, ...
    MiniBatchSize=miniBatchSize, ...
    ValidationPatience=30, ...
    L2Regularization=1e-2,...%10e-3, ...
    OutputNetwork='best-validation-loss');

[net,info] = trainNetwork(train_imds_aug,lgraph,options);
save(mdlName,"net");
end

% class => 8 - NC, 9 - G3, 10 - G4, 11 - G5
function [train_imds,test_imds,train_imds_aug,test_imds_aug,numTrainImages,numTestImages] = prepare_imds_training(trainData,testData,patch_size,miniBatchSize,class)
% Prepare Training and Validation datasets
[numTrainImages,~] = size(trainData);
[numTestImages,~] = size(testData);

% Define the image augmentation scheme.
imageAugmenter = imageDataAugmenter( ...
    RandXTranslation=[-50 50], ...
    RandYTranslation=[-50 50], ...
    RandXScale=[0.8, 1.2], ...    
    RandYScale=[0.8, 1.2], ...    
    RandRotation=[-45,45], ...
    RandYReflection=true, ...
    RandXReflection=true);

% Create the datastores
train_imds = imageDatastore(trainData.image_name);
test_imds = imageDatastore(testData.image_name);

% class => 8 - NC, 9 - G3, 10 - G4, 11 - G5 => [1:4] => [8:11]
train_imds.Labels = categorical(1+(class-1)*double(table2array(trainData(:,class+7))>0.1));
test_imds.Labels = categorical(1+(class-1)*double(table2array(testData(:,class+7))>0.1));

train_imds_aug = augmentedImageDatastore(patch_size,train_imds, ...
        DataAugmentation=imageAugmenter);
train_imds_aug.MiniBatchSize = miniBatchSize;

test_imds_aug = augmentedImageDatastore(patch_size,test_imds, ...
        DataAugmentation=imageAugmenter);
test_imds_aug.MiniBatchSize = miniBatchSize;
end

% class => 8 - NC, 9 - G3, 10 - G4, 11 - G5
function [test_imds_aug,Labels,Multi_Labels] = prepare_imds_testing(testData,patch_size)
% Prepare Testing dataset
[numTestImages,~] = size(testData);
% Create the datastore
test_imds = imageDatastore(testData.image_name);
Labels = categorical(vec2ind(table2array(testData(:,2:5))')');
Multi_Labels = double(table2array(testData(:,9:11))>0.01);
test_imds_aug = augmentedImageDatastore(patch_size,test_imds);
end

% -----------------------------------------------
% Evaluate
% -----------------------------------------------
function evaluate_detector(k, experiment, class_names, dataset_partition, testData)
fprintf('\n\nEvaluating Set: %s\n',char(dataset_partition(k)));
mdlName = ['GG_Classification_',char(class_names(2)),'_',char(dataset_partition(k)),'_ResNet18_Exp',num2str(experiment,'%.2d')];
% mdlName = ['GG_Classification_',char(class_names(2)),'_',char(dataset_partition(k)),'_InceptionV3_Exp',num2str(experiment,'%.2d')];

load(mdlName,"net");
net2 = net;
mdlName = ['GG_Classification_',char(class_names(3)),'_',char(dataset_partition(k)),'_ResNet18_Exp',num2str(experiment,'%.2d')];
% mdlName = ['GG_Classification_',char(class_names(3)),'_',char(dataset_partition(k)),'_InceptionV3_Exp',num2str(experiment,'%.2d')];
load(mdlName,"net");
net3 = net;
mdlName = ['GG_Classification_',char(class_names(4)),'_',char(dataset_partition(k)),'_ResNet18_Exp',num2str(experiment,'%.2d')];
% mdlName = ['GG_Classification_',char(class_names(4)),'_',char(dataset_partition(k)),'_InceptionV3_Exp',num2str(experiment,'%.2d')];
load(mdlName,"net");
net4 = net;

patch_size = [224 224];
[test_imds_aug,Labels,Multi_Labels] = prepare_imds_testing(testData,patch_size);
scores2 = net2.predict(test_imds_aug);
scores3 = net3.predict(test_imds_aug);
scores4 = net4.predict(test_imds_aug);
scores_combined = [scores2(:,2),scores3(:,2),scores4(:,2)];

% actualLabels = double(Labels);
% [m,predictedLabels] = max(scores_combined,[],2);
% predictedLabels = predictedLabels + 1;
% predictedLabels(m<0.5) = 1;
% % Calculate accuracy
% correctPredictions = sum(predictedLabels == actualLabels);
% totalPredictions = numel(actualLabels);
% accuracy = correctPredictions / totalPredictions;
% fprintf('Accuracy: %.2f%%\n', accuracy * 100);
% threshold = 0.5;
% Use a different threshold for each classifier (optimal threshold found
% using best point (F1-score)
thres1 = 0.5;
thres2 = 0.5;
thres3 = 0.5;

% thres1 = 0.748;
% thres2 = 0.3886;
% thres3 = 0.3886;

% keyboard
metrics_G3 = evaluate_classifer(Multi_Labels(:,1),scores_combined(:,1),thres1);
metrics_G4 = evaluate_classifer(Multi_Labels(:,2),scores_combined(:,2),thres2);
metrics_G5 = evaluate_classifer(Multi_Labels(:,3),scores_combined(:,3),thres3);
% metrics_G3.optimal_threshold
% metrics_G4.optimal_threshold
% metrics_G4.optimal_threshold


%Handle NC class separately
NC_label = double(sum(Multi_Labels,2)==0);
predictions = [scores_combined(:,1)>metrics_G3.optimal_threshold,scores_combined(:,2)>metrics_G4.optimal_threshold,scores_combined(:,3)>metrics_G4.optimal_threshold];
NC_predictions = sum(predictions,2)==0;
% All_predictions = [predictions, NC_predictions];
confusionMatrix_NC = confusionmat(double(NC_predictions), NC_label);
Total_Pos = confusionMatrix_NC(2, 1) + confusionMatrix_NC(2, 2);
Total_Neg = confusionMatrix_NC(1, 1) + confusionMatrix_NC(1, 2);
% Total = Total_Pos + Total_Neg;
TP = confusionMatrix_NC(2, 2);
TN = confusionMatrix_NC(1, 1);
FP = confusionMatrix_NC(1, 2);
FN = confusionMatrix_NC(2, 1);
precision = TP / (TP + FP);
recall = TP / (TP + FN);
F1_NC = 2 * (precision * recall) / (precision + recall);
Acc_NC = (TP + TN)/(Total_Pos+Total_Neg);

Acc = mean([metrics_G3.accuracy, metrics_G4.accuracy, metrics_G5.accuracy, Acc_NC])*100;
fprintf('Accuracy: %.2f, F1-Avg: %.2f, F1-NC:%.2f, F1-G3:%.2f, F1-G4:%.2f, F1-G5:%.2f, K: %.2f\n', Acc, mean([metrics_G3.F1, metrics_G4.F1, metrics_G5.F1, F1_NC]),F1_NC,metrics_G3.F1, metrics_G4.F1, metrics_G5.F1, mean([metrics_G3.kappa_score, metrics_G4.kappa_score, metrics_G5.kappa_score]));
% figure;
% plot(X, Y);
% xlabel('Recall');
% ylabel('Precision');
% title('Precision-Recall Curve');
% grid on;

end


function visual_evaluation(net,testData)
% Label Definitions: [1 2 3 4 5] => [NC GG3 GG4 GG5 G4C] C-cribriform
% Not using G4C at the moment because masks don't have it in the new
% dataset!
figure
[numTestImages,~] = size(testData);
for i=1:numTestImages
    i
    img = imread(testData.image_name{i});
    mask = imread(testData.mask_name{i});
    h = hist(double(mask(:)),[0:5]);
    h = h/sum(h);
    h = [h(1) h(4:6)];
    % [maskLabel,h] = getLabel_from_mask(mask);
    % originalLabel = Labels(i);
    % 
    img_preprocess = imresize(img,[224 224]);
    predictedScores = net.predict(img_preprocess)
    % [m,predictedLabel] = max(predictedScores,[],2);
    % % keyboard
    % 
    if(predictedScores(2)>predictedScores(1))
        subplot(2,2,1)
        imshow(img)
        subplot(2,2,2)
        imshow(uint8(double(mask)*50))
        subplot(2,2,3)
        bar([1:4],h)
        subplot(2,2,4)
        bar([1:2],predictedScores)
        % title([num2str(double(originalLabel)),', ',num2str(maskLabel)])
        % maskLabel
        % originalLabel
        % predictedLabel
        % predictedScores
        pause
    end
end
end

% -----------------------------------------------
% Done! Give yourself a pat on the back...
% -----------------------------------------------
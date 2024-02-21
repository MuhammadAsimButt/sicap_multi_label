function lgraph = customize_ResNet18(input_size,output_type,output_size,load_weights,classNames,classWeights)
% input_size = [224,224];
% load_weights = 1;
% output_type = 2;%0-Regression, 1-Classification, 2- Multi-label Classification
% output_size = 4;
% classNames = ["1" "2" "3" "4"];

if(load_weights)
    lgraph = layerGraph(resnet18);
else
    lgraph = resnet18('Weights','none');
end

inputLayer = imageInputLayer([input_size,3], Normalization="rescale-symmetric", Min=0, Max=255, Name="ImageInputLayer");
lgraph = replaceLayer(lgraph,"data",inputLayer);
% analyzeNetwork(lgraph)
% return

% %Remove some layers 
% lgraph = removeLayers(lgraph,"res5a_branch2a");
% lgraph = removeLayers(lgraph,"bn5a_branch2a");
% lgraph = removeLayers(lgraph,"res5a_branch2a_relu");
% lgraph = removeLayers(lgraph,"res5a_branch2b");
% lgraph = removeLayers(lgraph,"bn5a_branch2b");
% lgraph = removeLayers(lgraph,"res5a");
% lgraph = removeLayers(lgraph,"res5a_relu");
% lgraph = removeLayers(lgraph,"res5a_branch1");
% lgraph = removeLayers(lgraph,"bn5a_branch1");
% lgraph = removeLayers(lgraph,"res5b_branch2a");
% lgraph = removeLayers(lgraph,"bn5b_branch2a");
% lgraph = removeLayers(lgraph,"res5b_branch2a_relu");
% lgraph = removeLayers(lgraph,"res5b_branch2b");
% lgraph = removeLayers(lgraph,"bn5b_branch2b");
% lgraph = removeLayers(lgraph,"res5b");
% lgraph = removeLayers(lgraph,"res5b_relu");
% 
% lgraph = removeLayers(lgraph,"res4a_branch2a");
% lgraph = removeLayers(lgraph,"bn4a_branch2a");
% lgraph = removeLayers(lgraph,"res4a_branch2a_relu");
% lgraph = removeLayers(lgraph,"res4a_branch2b");
% lgraph = removeLayers(lgraph,"bn4a_branch2b");
% lgraph = removeLayers(lgraph,"res4a");
% lgraph = removeLayers(lgraph,"res4a_relu");
% lgraph = removeLayers(lgraph,"res4a_branch1");
% lgraph = removeLayers(lgraph,"bn4a_branch1");
% lgraph = removeLayers(lgraph,"res4b_branch2a");
% lgraph = removeLayers(lgraph,"bn4b_branch2a");
% lgraph = removeLayers(lgraph,"res4b_branch2a_relu");
% lgraph = removeLayers(lgraph,"res4b_branch2b");
% lgraph = removeLayers(lgraph,"bn4b_branch2b");
% lgraph = removeLayers(lgraph,"res4b");
% lgraph = removeLayers(lgraph,"res4b_relu");
% % 
% lgraph = removeLayers(lgraph,"res3a_branch2a");
% lgraph = removeLayers(lgraph,"bn3a_branch2a");
% lgraph = removeLayers(lgraph,"res3a_branch2a_relu");
% lgraph = removeLayers(lgraph,"res3a_branch2b");
% lgraph = removeLayers(lgraph,"bn3a_branch2b");
% lgraph = removeLayers(lgraph,"res3a");
% lgraph = removeLayers(lgraph,"res3a_relu");
% lgraph = removeLayers(lgraph,"res3a_branch1");
% lgraph = removeLayers(lgraph,"bn3a_branch1");
% lgraph = removeLayers(lgraph,"res3b_branch2a");
% lgraph = removeLayers(lgraph,"bn3b_branch2a");
% lgraph = removeLayers(lgraph,"res3b_branch2a_relu");
% lgraph = removeLayers(lgraph,"res3b_branch2b");
% lgraph = removeLayers(lgraph,"bn3b_branch2b");
% lgraph = removeLayers(lgraph,"res3b");
% lgraph = removeLayers(lgraph,"res3b_relu");
% 
% lgraph = connectLayers(lgraph,"res3b_relu","pool5");

% %Modify the head
if(output_type==1)
    fclayer = fullyConnectedLayer(output_size,'Name','fc4');
    fclayer = setLearnRateFactor(fclayer,"Weights",10);
    lgraph = replaceLayer(lgraph,"fc1000",fclayer);
    % classWeights = [0.25 0.25 0.25 0.25];
    
    cl = classificationLayer( ...
        'Classes',classNames, ...
        'ClassWeights',classWeights, ...
        'Name','output');
    lgraph = replaceLayer(lgraph,"ClassificationLayer_predictions",cl);
    
    %Add a dropout layer
    dropLayer = dropoutLayer(0.001);
    lgraph = addLayers(lgraph,dropLayer);
    lgraph = disconnectLayers(lgraph,"pool5","fc4");
    lgraph = connectLayers(lgraph,"pool5","dropout");
    lgraph = connectLayers(lgraph,"dropout","fc4");

elseif(output_type==0)
    fc10 = fullyConnectedLayer(10,'Name','fc10');
    lgraph = replaceLayer(lgraph,"fc1000",fc10);
    lgraph = removeLayers(lgraph,"prob");
    lgraph = removeLayers(lgraph,"ClassificationLayer_predictions");
    sig1 = sigmoidLayer(Name="sigmoid1");
    lgraph = addLayers(lgraph,sig1);
    lgraph = connectLayers(lgraph,"fc10","sigmoid1");
    fc4 = fullyConnectedLayer(output_size,'Name','fc4');
    lgraph = addLayers(lgraph,fc4);
    lgraph = connectLayers(lgraph,"sigmoid1","fc4");    
    sig2 = sigmoidLayer(Name="sigmoid2");
    lgraph = addLayers(lgraph,sig2);
    lgraph = connectLayers(lgraph,"fc4","sigmoid2");    
    rl = regressionLayer('Name',"Regression_Output");
    lgraph = addLayers(lgraph,rl);
    lgraph = connectLayers(lgraph,"sigmoid2","Regression_Output");
elseif(output_type==2)
    fc = fullyConnectedLayer(output_size,'Name','fc');
    lgraph = replaceLayer(lgraph,"fc1000",fc);
    newActivationLayer = sigmoidLayer(Name="sigmoid");
    lgraph = replaceLayer(lgraph,"prob",newActivationLayer);
    newOutputLayer = CustomXEntropyLossLayer("output");
    lgraph = replaceLayer(lgraph,"ClassificationLayer_predictions",newOutputLayer);
end
% analyzeNetwork(lgraph)
end






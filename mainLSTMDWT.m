% This code is written by Ali Ibrahim (aibrahim2014@fau.edu)
%Under supervision of Dr.Hanqi Zhuang and Dr.Laurent Cherubin
% This code is the example of training CNN by using DWT features, we used
% one file for each class as an example and to make data for training ,we
% duplicate each file 1000 times because we didn't upload the data. So, you
% can use this example to train your data.. x1 is red hind,x2 is nassau,x3
% is yellow fin,x4 is black grouper,and x5 is background....This CNN is 1D
% CNN and the sampling rate of our datasets is 10KHz, so you maybe try
% different structure of CNN to find best hyperparameters.If you have any
% question or need help, please don't hestitate to contact me
% ,(aibrahim2014@fau.edu)
clc;
clear;
close all;
load x1;
load x2;
load x3;
load x4;
load x5;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% In this cwe use only five samples as example to train the network.To
%%% make it work , we extend 
m=1;
for i=1:1000
    [cA,cD] = dwt(x1,'db4');
    [cA1,cD1] = dwt(cA,'db4');
    [cA2,cD] = dwt(cA1,'db4');
    XTrain{m}=cA2;% extract 3rd approximation from the Grouper sounds, You can Try 2nd level or third levels
    YTrain(m)=1;
    m=m+1;
end
for i=1:1000
    [cA,cD] = dwt(x2,'db4');
    [cA1,cD1] = dwt(cA,'db4');
    [cA2,cD] = dwt(cA1,'db4');
     XTrain{m}=cA2; % extract 3rd approximation from the Grouper sounds, You can Try 2nd level or third levels
    YTrain(m)=2;
    m=m+1;
end
for i=1:1000
    [cA,cD] = dwt(x3,'db4');
    [cA1,cD1] = dwt(cA,'db4');
    [cA2,cD] = dwt(cA1,'db4');
     XTrain{m}=cA2; % extract 3rd approximation from the Grouper sounds, You can Try 2nd level or third levels
    YTrain(m)=3;
    m=m+1;
end
for i=1:1000
    [cA,cD] = dwt(x4,'db4');
    [cA1,cD1] = dwt(cA,'db4');
    [cA2,cD] = dwt(cA1,'db4');
       XTrain{m}=cA2; % extract 3rd approximation from the Grouper sounds, You can Try 2nd level or third levels
    YTrain(m)=4;
    m=m+1;
end
for i=1:1000
    [cA,cD] = dwt(x5,'db4');
    [cA1,cD1] = dwt(cA,'db4');
    [cA2,cD] = dwt(cA1,'db4');
      XTrain{m}=cA2; % extract 3rd approximation from the Grouper sounds, You can Try 2nd level or third levels
    YTrain(m)=5;
    m=m+1;
end
YTrain = categorical( YTrain');
%%%%%%%%%%%%%%% Now divide the data to the training and Testing
idx = randperm(length(XTrain),1000);% choose randomly 1000 samples index for testing
XValidation = XTrain(idx);
XTrain(idx) = [];% Removing the Testing samples from the Training data
YValidation = YTrain(idx);% chosing The Testing samples Labels
YTrain(idx) = [];% Removing Testing labels 
%%%%%%%%%%%%%%%%%
inputSize = 2506;
numLayerRange = 1; % Nimber of LSTM layers , In this example only one LSTM layer
numHiddenUnits = 500;
numClasses = 5;
%%%%%%%%%%%%%%%%%%%%%
inLayers = [sequenceInputLayer(inputSize)];
    numLayers = numLayerRange;    
    for i = 1:numLayers
        midLayers = [
           lstmLayer(numHiddenUnits,'OutputMode','last')
        ];
        inLayers = cat(1, inLayers, midLayers);
    end
    
    outLayers = [
        fullyConnectedLayer(numClasses, 'name', 'out')
        softmaxLayer
        classificationLayer
    ];
    net = cat(1, inLayers, outLayers);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

maxEpochs = 20;
miniBatchSize = 200;

options = trainingOptions('adam', ...
    'ExecutionEnvironment','cpu', ...
    'GradientThreshold',1, ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest', ...
    'Shuffle','never', ...
    'Verbose',0, ...
    'Plots','training-progress');

net = trainNetwork(XTrain,YTrain,net,options);
%%%%%%%%%%
temp= classify(net,XValidation,'MiniBatchSize',miniBatchSize,'SequenceLength','longest');
Acc = sum(temp == YValidation)/numel(YValidation);
fprintf("%.2f ", Acc * 100);

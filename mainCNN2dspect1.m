
% This code is written by Ali Ibrahim (aibrahim2014@fau.edu)
%Under supervision of Dr.Hanqi Zhuang and Dr.Laurent Cherubin
% This code is the example of training CNN by using Spectrogram images of the sounds after decimation by factor 10, we used
% one file for each class as an example and to make data for training ,we
% duplicate each file 1000 times because we didn't upload the data. So, you
% can use this example to train your data.. x1 is red hind,x2 is nassau,x3
% is yellow fin,x4 is black grouper,and x5 is background....This CNN is 2D
% CNN and the sampling rate of our datasets is 10KHz, so you maybe try
% different structure of CNN to find best hyperparameters.If you have any
% question or need help, please don't hestitate to contact me
% ,(aibrahim2014@fau.edu)
clc;
clear;
close all;
addpath('data');
addpath('Spectrofeatures')
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
    XTrain(:,:,:,m)=spectroo(decimate(x1,10),1000);% extract spectrogram features and convert them to RGB image from the Grouper sounds
    YTrain(m)=1;
    m=m+1;
end
for i=1:1000
    XTrain(:,:,:,m)=spectroo(decimate(x2,10),1000);% extract spectrogram features and convert them to RGB image from the Grouper sounds
    YTrain(m)=2;
    m=m+1;
end
for i=1:1000
    XTrain(:,:,:,m)=spectroo(decimate(x3,10),1000);% extract spectrogram features and convert them to RGB image from the Grouper sounds
    YTrain(m)=3;
    m=m+1;
end
for i=1:1000
    XTrain(:,:,:,m)=spectroo(decimate(x4,10),1000);% extract spectrogram features and convert them to RGB image from the Grouper sounds
    YTrain(m)=4;
    m=m+1;
end
for i=1:1000
    XTrain(:,:,:,m)=spectroo(decimate(x5,10),1000);% extract first approximation from the Grouper sounds, You can Try 2nd level or third levels
    YTrain(m)=5;
    m=m+1;
end
YTrain = categorical( YTrain');
%%%%%%%%%%%%%%% Now divide the data to the training and Testing
idx = randperm(size(XTrain,4),1000);% choose randomly 1000 samples index for testing
XValidation = XTrain(:,:,:,idx);
XTrain(:,:,:,idx) = [];% Removing the Testing samples from the Training data
YValidation = YTrain(idx);% chosing The Testing samples Labels
YTrain(idx) = [];% Removing Testing labels 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

numClasses = 5;% Number of classes
numLayerRange = [1 3];% NUmber of convolutional block range 
numNeuronRange = [16 256];% Number of filters in each of convolutional layer, here is randomly chosen
numNetworks = 1;
inputSize = size(XTrain(:,:,:,1));
windowSize = 5;% filter size may be you can change it to 3 or 11 and so on
fprintf("Generating Networks: ");
%%%%%%%%%% Here we will generate the model
inLayers = [imageInputLayer(inputSize)];
    numLayers = randi(numLayerRange);
    numNeurons = sort(randi(numNeuronRange, numLayers, 1), 'descend');
    
    for i = 1:numLayers
        midLayers = [
            convolution2dLayer(windowSize, numNeurons(i))
            batchNormalizationLayer
            reluLayer
            maxPooling2dLayer(windowSize, 'stride', 2)
        ];
        inLayers = cat(1, inLayers, midLayers);
    end
    
    outLayers = [
        fullyConnectedLayer(numClasses, 'name', 'out')
        softmaxLayer
        classificationLayer
    ];
    
    net = cat(1, inLayers, outLayers);
%%%%%%%%%%


fprintf("Done\n");

%%%% Training Phase
    miniBatchSize = 64;
    initialLearnRate = 1e-1 * miniBatchSize/256;
    options = trainingOptions('sgdm', ...
        'MiniBatchSize',miniBatchSize, ....
        'Verbose',false, ...
        'InitialLearnRate',initialLearnRate, ...
        'L2Regularization',1e-10, ...
        'MaxEpochs',5, ...
        'Shuffle','every-epoch', ...
        'ValidationPatience',Inf, ...
        'LearnRateSchedule','piecewise', ...
        'LearnRateDropFactor',0.1, ...
        'LearnRateDropPeriod',25,'Plots','training-progress');
    
    net = {trainNetwork(XTrain,YTrain, net, options)};
    fprintf("Done\n");

% Testing 
    temp = classify(net,XValidation);
    Acc = sum(temp == YValidation)/numel(YValidation);
    fprintf("%.2f ",Acc * 100);
  


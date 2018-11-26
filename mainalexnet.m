% This code is written by Ali Ibrahim (aibrahim2014@fau.edu)
%Under supervisionn of Dr.Hanqi Zhuang and Dr.Laurent Cherubin
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
addpath('data')
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
     u=decimate(x1,10);
    XTrain(:,:,:,m)=imresize(spectroo(u,1000),[227 227]);% Convert the spectrogram to image and resize it to fit with alexnet
    YTrain(m)=1;
    m=m+1;
end
for i=1:1000
     u=decimate(x2,10);
    XTrain(:,:,:,m)=imresize(spectroo(u,1000),[227 227]);% Convert the spectrogram to image and resize it to fit with alexnet
    YTrain(m)=2;
    m=m+1;
end
for i=1:1000
     u=decimate(x3,10);
    XTrain(:,:,:,m)=imresize(spectroo(u,1000),[227 227]);% Convert the spectrogram to image and resize it to fit with alexnet
    YTrain(m)=3;
    m=m+1;
end
for i=1:1000
     u=decimate(x4,10);
    XTrain(:,:,:,m)=imresize(spectroo(u,1000),[227 227]);% Convert the spectrogram to image and resize it to fit with alexnet
    YTrain(m)=4;
    m=m+1;
end
for i=1:1000
     u=decimate(x5,10);
    XTrain(:,:,:,m)=imresize(spectroo(u,1000),[227 227]);% Convert the spectrogram to image and resize it to fit with alexnet
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Tis code uses alexnet, you have to install
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% alex net and we do Transfer Learning
tempLayers = alexnet;
transferLayers = tempLayers.Layers(1:end-3);
layers = [
    transferLayers
    fullyConnectedLayer(5)
    softmaxLayer
    classificationLayer];
options = trainingOptions('sgdm', ...
        'MaxEpochs', 200,...
        'ValidationFrequency', 5000,...
        'Verbose', false,...
        'Plots', 'training-progress',...
        'ValidationPatience', Inf);
   
 net = trainNetwork(XTrain,YTrain, layers, options);
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   
   %%%Testing
   temp = classify(net,XValidation);
   Acc = sum(temp == YValidation)/numel(YValidation);
   fprintf("%.2f ",Acc * 100);
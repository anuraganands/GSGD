% Change the filenames if you've saved the files under different names
% On some platforms, the files might be saved as 
% train-images.idx3-ubyte / train-labels.idx1-ubyte
clc
clear all
images = loadMNISTImages('train-images.idx3-ubyte');
labels = loadMNISTLabels('train-labels.idx1-ubyte');
 
% We are using display_network from the autoencoder code
addpath(genpath('starter'));
display_network(images(:,1:2)); % Show the first 100 images
disp(labels(1:2));
disp(reshape(images(:,1),[28 28]))
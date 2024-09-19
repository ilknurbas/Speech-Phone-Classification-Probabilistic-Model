%% Goals: 
% Implement Gaussian classifier for phones
% Implement GMM classifier for phones
%
% Data: Librispeech dev-clean

clear all

curdir = fileparts(which('E3_main.m'));
addpath([curdir '/aux_scripts/']);
addpath([curdir '/sap-voicebox/voicebox/']);
curdir

%% Step 0) Data pre-processing (pre-provided scripts)

% Load LibriSpeech (CHANGE TO YOUR OWN PATH)

% datapath = '/Users/rasaneno/speechdb/LibriSpeech/';
datapath = '/Users/ilknurbas/Documents/MATLAB/Speech Processing/#3/LibriSpeech/';

% Load LibriSpeech MFCC feature vectors and phone labels, using 80% of
% data as training data and rest for testing. MFCC vectors are 39 dim
% features with 13 static, 13 time derivative, and 13 second time
% derivative ("deltas and deltadeltas"), and using mean and variance
% normalization across utterances.
[x_train,labels_train,x_test,labels_test,unique_phones] = loadLibriData_E3(datapath,0.8);



%% Actual Exercise 3 tasks start here
%% Step 1) Model each class with a multivariate normal distribution
% For this, you will need:
%   1) a routine to train a GM, implemented in trainGaussian.m
%   2) a routine to check probability of a class, given the model and a
%   sample x, i.e., P(class | x, GM), implemented in testGaussian.m

GM = trainGaussian(x_train,labels_train);

[loglik,predicted_labels] = testGaussian(x_test,GM);

% Evaluate (code pre-provided)
[accuracy,UAR,confmat] = evaluateClassification(predicted_labels,labels_test);

fprintf('Classification accuracy: %0.2f%%.\n',accuracy); % 42.40%.

% Plotting to be included to the report
printConfusionMatrix(confmat,unique_phones);

printModelParams(GM);


%% Step 2) Model each class with a GMM
% As above, the idea is to train a separate GMM for each phone class, 
% so you will need:
%   1) a routine to train a GMM, implemented in trainGMM.m
%   2) a routine to check probability of a class, given the model and a
%   sample x, i.e., P(class | x, GMM), implemented in testGMM.m
% Please see task instruction document for more details. 


% GMM parameters 
use_cov = 1;    % use full covariance matrices?
n_comps = 8;    % number of Gaussians per mixture
max_iter = 30;  % max number of training iterations (EM iterations)

% Train the model
GMM = trainGMM(x_train,labels_train,n_comps,use_cov,max_iter);

% Test the model
[loglik,predicted_labels] = testGMM(x_test,GMM);

% Evaluate
[accuracy,UAR,confmat] = evaluateClassification(predicted_labels,labels_test);

fprintf('Classification accuracy: %0.2f%%.\n',accuracy);

printConfusionMatrix(confmat,unique_phones);

printLogLiks(GMM);

%% Application of Gaussian or GMM model to 251-136532-0016.flac and visualization of the output

[x,fs] = audioread([curdir '/extra_data/251-136532-0016.flac']);

MFCC_251 = getMFCCs(x,fs,0.02,0.01,12,1,1,1);

[loglik_251,predicted_labels_251] = testGMM(MFCC_251,GMM);

TG = readTextGrid([curdir '/extra_data/251-136532-0016_reference.TextGrid']);

drawExampleAudio(x,fs,predicted_labels_251,TG,unique_phones)








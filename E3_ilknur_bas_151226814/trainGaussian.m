function GM = trainGaussian(x_train,labels_train,usecov)
% function GM = trainGaussian(x_train,labels_train,usecov)
%
% Trains a multivariate Gaussian model for each data class in
% [labels_train] using samples in [x_train]
%
% Inputs:   
%       x_train:        training data (N x fdim)
%       labels_train:   corresponding class labels [1, 2, ..., N]
%       usecov:         use full covariance matrix? (0/1; default = 1)
%
% Outputs:
%       GM:              struct containing the model for each class

if nargin <3
    usecov = 1;
end

N_classes = max(labels_train);  % total number of classes

% Since models are going to be independent, let's store the means and
% variances into cell arrays for code transparency. 

GM = struct();
GM.info = 'Multivariate Gaussian model for each phone class';

GM.mean = cell(N_classes,1);     % means of class-specific Gaussians
GM.sigma = cell(N_classes,1);    % covariances of class-specific Gaussians

fdim = size(x_train,2); % feature dimensionality

for class = 1:N_classes % Iterate through classes
    
    x_class = x_train(labels_train == class,:);  % select samples from the class    
    
    % GM.mean{class} = ?   % estimate mean of the class 
    % mean of the MFCC feature vectors for samples
    GM.mean{class} = mean(x_class);
        
    
    if(~usecov) % (optional) create a diagonal covariance matrix of size fdim x fdim
        % GM.sigma{class} = ?  % estimate diagonal covariance for current class       
        
    else % full covariance matrix of size fdim x fdim
        % GM.sigma{class} = ?   % estimate covariance for current class
        GM.sigma{class} = cov(x_class);
    end
        
    % Make sure there are no zero variances on diagonal or NaN variances
    % in general (e.g., due to numerical problems or sample insufficiency)    
    GM.sigma{class} = GM.sigma{class}+diag(ones(fdim,1)).*1e-12;
    GM.sigma{class}(isnan(GM.sigma{class})) = 0;
        
end

% Calculcate class frequencies in training data, and from there class prior
% probabilities (as a vector with N_classes dimensions).

% class_frequencies = ?
class_frequencies = histcounts(labels_train,1:N_classes+1);
size(class_frequencies); %  1    42

% GM.class_priors = ?
GM.class_priors = class_frequencies/size(x_train,1);
size(GM.class_priors); %  1    42
display(sum(GM.class_priors));  % sums up to 1







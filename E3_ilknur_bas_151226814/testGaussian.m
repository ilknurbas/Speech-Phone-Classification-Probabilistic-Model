function [loglik,predicted_labels] = testGaussian(x_test,GM)
% function loglik = testGaussian(x_test,GM)
% 
% Estimates log-likelihoods of data in x_test for each Gaussian model
% stored in struct GM.
%
% Inputs:   
%       x_test:     data to test
%       GM:          model struct for Gaussian models
%
% Outputs:
%       loglik:     "log-likelihood", aka. logarithm of pdf value for each
%                    model and each sample (N_samples x N_classes)
% Hints:    
%       H1: Normal multivariate gaussian probability density returns
%       likelihoods in linear domain, but logs of likelihoods
%       ("log-likelihoods") are often more convenient. This is since the 
%       probabilities can become extremely small, causing problems with 
%       numerical precision. 
%       
%       

N_classes = length(GM.mean);  % total number of classes

% Since models are going to be independent, let's store the means and
% variances into cell arrays for code transparency. 

fdim = size(x_test,2); % feature dimensionality
N_samples = size(x_test,1);

loglik = zeros(N_samples,N_classes); % log-likelihoods for each sample and class

for class = 1:N_classes % Iterate through classes
        
    mu = GM.mean{class}';       % mean of the current class
    sigma = GM.sigma{class};   % covariance matrix of the current class
    size(sigma);  % 36    36
    size(mu);  %  36     1
    
     % Iterate through each sample in x_test
    for sample = 1:size(x_test,1)
        
        x = x_test(sample,:)'; % current input sample to the classifier
        
        % Implement log-likelihood calculation from probability density of a Gaussian
        % (see Eqs. 1.1 and 1.6 in the instructions) to get log-likelihood of
        % the given sample for the given class.

        % loglik(sample,class) = ?        
        loglik(sample,class) = log(exp(-0.5*(x-mu)'*(sigma^(-1))*(x-mu))/(sqrt(power(2*pi,fdim)*det(sigma))));
        
    end
    procbar(class,N_classes);
end


% Add the effect of class prior probability to the log likelihoods to get
% (unnormalized) logarithmic class posteriors (see Eqs. 1.4 and 1.6)

% class_posteriors = ?
size(loglik); %  388344          42
size(log(GM.class_priors)); % 1    42
class_posteriors = loglik + log(GM.class_priors);


% Determine predicted class label for each sample as the class with the
% highest class posterior
[~,predicted_labels] = max(class_posteriors,[],2);





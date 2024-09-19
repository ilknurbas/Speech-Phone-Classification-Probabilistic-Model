function [loglik,predicted_labels] = testGMM(x_test,GMM)
% function [loglik,predicted_labels] = testGMM(x_test,GMM)
% 
% Estimates log-likelihoods of data in x_test for each Gaussian model
% stored in struct M.
%
% Inputs:   
%       x_test:             data to test
%       M:                  model struct for Gaussian models
%
% Outputs:
%       loglik_class:       log-likelihood of each class for each sample
%       predicted_labels:   predicted class of each sample
%       comp_probs:         GMM component probabilities for each class
%                           and sample


N_classes = length(GMM.means);  % total number of classes
N_samples = size(x_test,1); % 388344
fdim = size(x_test,2); % feature dimensionality 36

n_comps = length(GMM.weights{1});

loglik = zeros(N_samples,N_classes);

for class = 1:N_classes % Iterate through classes
    
    %% Expectation (E)-step
    % Calculate "probability" (pdf value) for each sample for each model
    % component
    
    comp_probs = zeros(size(x_test,1),n_comps);
    

    for comp = 1:n_comps
        comp_mean = GMM.means{class}(comp,:);
        comp_sigma = GMM.sigmas{class}(:,:,comp);
        comp_weight = GMM.weights{class}(comp);
       
        size(comp_mean); %   1    36
        size(comp_sigma); %   36    36
        size(comp_weight); %  1     1
        size(comp_probs(:,comp)); %  388344           1

        % TASK: get probability that a Gaussian component generated the
        % feature vectors in x_train
        % comp_probs(:,comp) = ?         (use Eq. 1.1 for each k)
        
        comp_probs(:,comp) = comp_weight* mvnpdf(x_test,comp_mean,comp_sigma);
    end
    
    % TASK: get log-likelihood of each class using the measured component
    % likelihoods.
    
    % loglik(:,class) = ?               (Eq. 2.1)
    loglik(:,class) = log(sum(comp_probs,2));

     
    procbar(class,N_classes); 
    
end

% Convert to data log-likelihoods to relative class likelihoods with the
% help of previously measured GMM.class_priors
% class_posteriors = ?
size(loglik); %  388344          42
size(log(GMM.class_priors)); % 1    42
class_posteriors = loglik + log(GMM.class_priors);

% Find the best model 
[~,predicted_labels] = max(class_posteriors,[],2);






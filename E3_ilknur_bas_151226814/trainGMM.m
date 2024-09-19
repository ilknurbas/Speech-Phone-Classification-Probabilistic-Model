function GMM = trainGMM(x_train,labels_train,n_comps,usecov,n_iters)
% function GMM = trainGMM(x_train,labels_train,n_comps,usecov,n_iters)
%
% Now, lets make a Gaussian Mixture Model (GMM) where each phone is modeled
% by several Gaussian components.
%

if nargin <3
    n_comps = 3;
end

if nargin <4
    usecov = 1;
end

if nargin <5
    n_iters = 20;
end

N_classes = max(labels_train);  % total number of classes

% Since models are going to be independent, let's store the means and
% variances into cell arrays for code transparency. 

GMM = struct();
GMM.info = 'Multivariate Gaussian mixture model for each phone class';

GMM.means = cell(N_classes,1);     % means of class-specific Gaussians
GMM.sigmas = cell(N_classes,1);    % covariances of class-specific Gaussians
GMM.weights = cell(N_classes,1);    % weights of class-specific Gaussians

fdim = size(x_train,2); % feature dimensionality

%% Initialize means and covariances randomly using data mean and variance as
%% the prior distribution
% (you don't need to edit this part, it will just provide initial value for
% weights, means, and covariances).

for class = 1:N_classes % Iterate through classes
    x_class = x_train(labels_train == class,:);  % select samples from the class
    
    mu_all = mean(x_class);
    cov_all = cov(x_class);
    
    GMM.means{class} = zeros(n_comps,fdim);
    GMM.sigmas{class} = zeros(fdim,fdim,n_comps);
    
    % Initialize component weights as 1/n_comp
    GMM.weights{class} = ones(n_comps,1)./n_comps;
    
    for comp = 1:n_comps
        % Sample mean from the data
        GMM.means{class}(comp,:) = mvnrnd(mu_all,cov_all);
        
        % Get covariance as full data covariance + some white noise
        if(usecov)
            tmp = randn(size(cov_all))*0.1;
            for j = 1:size(tmp,1)
               tmp(j,j) = abs(tmp(j,j)); % ensure positive diagonal variance
            end
            tmp = tmp*tmp'; % make symmetric
            GMM.sigmas{class}(:,:,comp) = cov_all+tmp;
        else
            tmp = rand(size(diag(cov_all)))*0.1;
            
            GMM.sigmas{class}(:,:,comp) = diag(diag(cov_all))+diag(tmp);            
        end
    end
end


LL = zeros(N_classes,100).*NaN;
%% Training part
for class = 1:N_classes % Iterate through phone classes
    x_class = x_train(labels_train == class,:);  % select samples from the class
      
    
    loglik = zeros(n_iters,1);
    for iter = 1:n_iters % Train for multiple epochs

        % Expectation (E)-step
        % Calculate likelihood (pdf value) for each sample for each model
        % component
        
        comp_likelihood = zeros(size(x_class,1),n_comps);
        
        for comp = 1:n_comps % iterate over Gaussians ("components")
            comp_mean = GMM.means{class}(comp,:);
            comp_sigma = GMM.sigmas{class}(:,:,comp);
            comp_weight = GMM.weights{class}(comp);           
            
            % Task: estimate likelihood for the current model component 
            % for all samples in x_class, taking the component weight into
            % account. You can use mvnpdf() for
            % multivariate gaussian probability density function or use
            % your own code from the Gaussian classifier.  
            % See numerator in Eq. (2.2) for formula.
            
            %comp_likelihood(:,comp) = ?  
            comp_likelihood(:,comp) = comp_weight * mvnpdf(x_class,comp_mean,comp_sigma);
        end
        
        % TASK: Calculate posterior probability that each component produced the
        % samples (add influence of denominator in Eq. 2.2 to complete Eq. 2.2 evaluation). 
        % Dimension of h should correspond to dimension of
        % comp_likelihood (samples x components).
                
        %h = ?
        % ./
        h = comp_likelihood./sum(comp_likelihood,2);
        
        % Calculate total data likelihood for this iteration 
        loglik(iter) = log(mean(sum(comp_likelihood,2)));
        
        % Maximization (M)-step:
        % Calculate new mean, sigma and weights
        
        for comp = 1:n_comps

            %% TASK: define here new component weights, means and covariance
            % parameters based on M-step of the EM-algorithm.
            
            %new_comp_weight = ?            % Eq. (2.3)
            %new_comp_mean = ?              % Eq. (2.4)
            %new_comp_sigma = ?             % Eq. (2.5)
            N_samples = size(x_class,1);

            new_comp_weight = sum(h(:,comp),1)/N_samples;
            new_comp_mean = sum(h(:,comp).*x_class,1)/sum(h(:,comp),1);
            new_comp_sigma = h(:,comp)'.* (x_class-GMM.means{class}(comp,:))'*(x_class - GMM.means{class}(comp,:))./sum(h(:,comp));
            
             
            %% Replace original parameters with the updated ones
            GMM.weights{class}(comp,:) = new_comp_weight;
            GMM.means{class}(comp,:) = new_comp_mean;
            GMM.sigmas{class}(:,:,comp) = new_comp_sigma+diag(ones(fdim,1)).*1e-6; % a trick to keep covariance matrices positive definite
                        
            % A hack to keep diagonal covariance model diagonal.
            if(~usecov)                 
                tmp = diag(GMM.sigmas{class}(:,:,comp));               
                GMM.sigmas{class}(:,:,comp) = zeros(size(GMM.sigmas{class}(:,:,comp)))+diag(tmp);
            end                      
        end
        
        fprintf('Class %d, log-lik (iter %d): %0.5f\n',class,iter,loglik(iter));
        
        % Stop training when log-likelihood improvement is smaller than
        % delta (1e-5).
        if(iter > 1)
            if(loglik(iter) <= loglik(iter-1)+1e-5)
                break;
            end
        end
        
    end
    GMM.LL(class,1:min(100,length(loglik))) = loglik(1:min(100,length(loglik)));  % store log-likelihoods for debugging
end


% TASK: Calculcate class priors (same as in the Gaussian model) to be used in
% Bayesian equation.

% class_frequencies = ?
class_frequencies = histcounts(labels_train,1:N_classes+1);
size(class_frequencies); % 1    42 

% GMM.class_priors = ?
GMM.class_priors = class_frequencies / size(x_train,1);
size(GMM.class_priors); % 1    42 
sum(GMM.class_priors); % 1




function k = variance_hard_threshold(X, threshold)

% Computes the number of principal components (PCs) 
% needed to reach a given cumulative variance threshold.
%
%
% Inputs:
%   X        - A numeric matrix of size (m x n), where m is the number of observations/time points
%              (rows) and n is the number of variables (columns). Prior mean centering and
%              some normalization are advisable.
%
%   threshold - A scalar between 0 and 100 representing the cumulative variance threshold.
%
% Output:
%   k        - A scalar indicating the number of principal components required to reach the threshold
%              for the cumulative variance explained.
%
%
% Author: [Francesco E. Vaccari, PhD]
% Date: [12/11/2024]

X = X - mean(X); %just to be sure, since it can disrupt results

% Perform PCA using svd (or pca function in MATLAB)
[~,~,~,~,explained,~] = pca(X);

% Calculate the cumulative variance explained
cumulative_variance = cumsum(explained);

% Find the number of components required to reach the threshold
k = min(find(cumulative_variance > threshold));

end

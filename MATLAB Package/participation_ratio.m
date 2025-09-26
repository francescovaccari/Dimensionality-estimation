function PR = participatio_ratio(X)

% participation_ratio - Calculates the participation ratio of the eigenspectrum 
% of the input matrix to assess the dimensionality of the data.
%
% This function computes the participation ratio (PR) of the eigenvalues from the 
% covariance matrix of the input data. The participation ratio is a measure of the 
% concentration of variance across the principal components, with higher values 
% indicating a more evenly distributed eigenspectrum, suggesting an higher dimensionality.
%
% Reference:
%   Gao, et al., 2017. "Participation ratio as a measure of dimensionality", 
%   https://www.biorxiv.org/content/10.1101/214262v2
%
% Inputs:
%   X        - A numeric matrix of size (m x n), where m is the number of observations/time points 
%              (rows) and n is the number of variables (columns). This matrix represents 
%              the data whose covariance matrix will be analyzed. Prior mean centering and
%              some normalization are advisable.
%
% Output:
%   PR       - A scalar value representing the participation ratio of the eigenvalues 
%              of the covariance matrix of the input data.
%
%
% Author: [Francesco E. Vaccari, PhD]
% Date: [12/11/2024]

X = X - mean(X); %just to be sure, since it can disrupt results

C = cov(X);
eigenvalues = eig(C);
eigenvalues = sort(eigenvalues,'descend');
PR = (sum(eigenvalues))^2 / sum(eigenvalues.^2);

end

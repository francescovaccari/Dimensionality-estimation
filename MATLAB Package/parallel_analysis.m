function k = parallel_analysis(X, quant, num_rep)

% parallel_analysis - Performs Parallel Analysis to estimate the number of significant 
% dimensions (principal components) in a given data matrix.
%
% This function compares the eigenvalues of the covariance matrix of the real data 
% with the eigenvalues obtained from shuffled versions of the data.
% Parallel analysi suggest to retain all dimensions which associated
% eigenvalue is greater than the qth quantile of the matched null
% distribution.
%
% Inputs:
%   X        - A numeric matrix of size (m x n), where m is the number of observations/time points 
%              (rows) and n is the number of variables (columns). This matrix contains 
%              the data to perform parallel analysis on. Prior mean centering and
%              some normalization are advisable.
%   quant    - A scalar between 0 and 1, specifying the quantile (e.g., 0.95) to use 
%              for determining the threshold from the shuffled eigenvalue distribution.
%   num_rep  - An integer specifying the number of shuffling iterations to generate 
%              the null distribution of eigenvalues (e.g., 250).
%
% Output:
%   k        - A scalar indicating the estimated number of significant dimensions 
%              based on the comparison of real and shuffled eigenvalues.
%
% Example usage:
%   k = parallel_analysis(X, 0.95, 250);
%
%   This would return the number of significant components/dimensions (k) based on 
%   a 95th percentile threshold from 250 shuffling iterations for matrix X.
%
% Author: [Francesco E. Vaccari, PhD]
% Date: [12/11/2024]


c = cov(X);
real_e = eig(c);
real_e = sort(real_e, 'descend')';

fake_e = nan(num_rep, size(X,2));
for r = 1:num_rep
    %% Shuffling along columns
    [m,n]=size(X);
    [~,new_rows]=sort(rand([m,n]));
    new_cols=repmat(1:n,m,1);
    Xshuffle = X(sub2ind([m,n],new_rows,new_cols));

    %% null eigenvalues distr
    c = cov(Xshuffle);
    tmp = eig(c);
    fake_e(r, :) = sort(tmp, 'descend')';
end

Q = quantile(fake_e,quant);

for k = 1:length(Q)
    if find(real_e(k) < Q(k))
        k = k-1;
        break
    end
end


end
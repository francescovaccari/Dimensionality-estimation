function k = singval_hard_threshold(X)
%--------------------------------------------------------------------------
% SINGVAL_HARD_THRESHOLD Estimates matrix rank using optimal singular value
% hard thresholding
%
% Implements the optimal hard threshold method for singular values as described
% in Donoho & Gavish (2014). The method determines the number of significant
% singular values by comparing them to a theoretically optimal threshold based
% on random matrix theory and the Marcenko-Pastur distribution.
%
% Reference:
%   Donoho, D. L., & Gavish, M. (2014). The optimal hard threshold for 
%   singular values is 4/sqrt(3). IEEE Transactions on Information Theory, 
%   60(8), 5040-5053.
%
% Syntax:
%   k = singval_hard_threshold(X)
%
% Input:
%   X - Data matrix (rows: observations / columns: variables)
%
% Output:
%   k - Estimated significant rank (number of significant singular values)
%
% Method:
%   1. Computes SVD of centered data matrix
%   2. Calculates optimal threshold coefficient based on matrix aspect ratio
%   3. Applies threshold at median(singular_values) * coefficient
%   4. Counts singular values above threshold
%
% Note: 
%   The method assumes noise is present and uses median singular value as
%   noise estimate. It is particularly effective when:
%   - The true rank is much smaller than matrix dimensions
%   - The noise level is unknown
%   - The matrix aspect ratio is not too large (beta = m/n ≤ 1)
%
% Author: Francesco E. Vaccari, PhD
% Date: September 24, 2025
%--------------------------------------------------------------------------

% Center the data
X = X - mean(X); 

% Get matrix dimensions
m = size(X,2);
n = size(X,1);

% Compute SVD
[U, S, V] = svd(X);
S = diag(S);

% Calculate matrix aspect ratio (beta)
beta = m/n;
if (beta > 1)
    warning('Aspect ratio of matrix X is incorrect (beta > 1). Setting it to 1 to proceed')
    beta = 1;
end

% Get optimal threshold coefficient using Donoho-Gavish formula
coef = optimal_SVHT_coef(beta,0);

% Count singular values above threshold
% Threshold is set at: median(singular_values) * optimal_coefficient
k = length(find(S >= (coef * median(S))));

end

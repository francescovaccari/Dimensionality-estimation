function [B] = soft_normalize(A, k, dim)
%--------------------------------------------------------------------------
% SOFT_NORMALIZE Soft normalization of matrices and vectors
%
% Normalizes data using a "soft" normalization approach that adds a constant to 
% the range in the denominator. This helps prevent extreme amplification of noise 
% when the range is very small. Particularly useful for neural data.
%
% Syntax:
%   B = soft_normalize(A, k)
%   B = soft_normalize(A, k, dim)
%
% Inputs:
%   A     - Input matrix or vector to be normalized (numeric array)
%   k     - Constant added to range in denominator
%   dim   - (Optional) Dimension for normalization:
%           1 = along rows (default)
%           2 = along columns
%           Not needed if A is a vector
%
% Output:
%   B     - Normalized matrix/vector, same size as input A
%           Formula: B = A / (range(A) + k)
%
% Example:
%   % Normalize a matrix along columns with k=5
%   X = rand(100,10);
%   Xnorm = soft_normalize(X, 5, 2);
%
%   % Normalize a vector
%   v = rand(100,1);
%   vnorm = soft_normalize(v, 5);
%
% Author: Francesco E. Vaccari, PhD
% Date: September 24, 2025
%--------------------------------------------------------------------------


% If A is a vector, handle it as a 1D input, no need for 'dim'
if isvector(A)
    B = A ./ (range(A) + k);
    return;
else

    % If A is a matrix and dim is not provided, set default value for dim (normalize along rows)
    if nargin < 2
        dim = 1;  % Default to normalize along rows
    end

    if dim == 2
        A = A';
    end

    for neu = 1:size(A, 1)
        resp = A(neu,:);
        B(neu,:) = resp ./ (range(resp) + k);
    end

    % If input was transposed, transpose the result back
    if dim == 2
        B = B';
    end
end

end

function [B] = soft_normalize(A, k, dim)
% SOFT_NORMALIZE Normalizes the input matrix or vector A using the formula:
%   norm_response = response / (range(response) + k)
% The normalization is applied along a specified dimension (rows or columns).
% If A is a vector, normalization is applied to the vector itself.
%
% Inputs:
%   A     - Input matrix or vector to be normalized (numeric array).
%   k     - A constant added to the range in the denominator (often equal to 5 for cortical firing rates).
%   dim   - (Optional) Dimension along which to normalize:
%          1 for normalization along rows (DEFAULT),
%          2 for normalization along columns.
%          If A is a vector, dim is not needed.
%
% Outputs:
%   B     - Normalized matrix or vector with the same size as A.
%


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

function k = kaiser_rule(X, varargin)
% Apply the Kaiser Rule to select number of PCs to retain based on
% eigenvalue threshold or explained variance (adaptation for data that
% are not z-scored)
%
% INPUT:
%   X           - Data matrix (each column is a variable, each row is
%               an observation/time point). It should mean-centered and normalized.
%   UseVariance - (optional) boolean flag to compute k based on explained variance
%
% OUTPUT:
%   k           - Number of PCs to retain
%
% Example usage:
%   k = kaiser_rule(X,'UseVariance',true);
%
%   This would return the number of significant components/dimensions (k) based on
%   the explained variance adaptation for matrix X.
%
% Author: [Francesco E. Vaccari, PhD]
% Date: [12/11/2024]


% Create an input parser to handle optional arguments
p = inputParser;

% Add a name-value pair argument for UseVariance, defaulting to false
addParameter(p, 'UseVariance', false, @(x) islogical(x) || ismember(x, [0, 1]));

% Parse the input arguments
parse(p, varargin{:});

% Get the value of UseVariance from the parsed inputs
UseVariance = p.Results.UseVariance;

X = X - mean(X); %just to be sure, since it can disrupt results

if UseVariance
    [~,~,~,~,explained,~] = pca(X);

    % Compute the variance explained by the original variables
    OrigVar = mean( var(X) / sum(var(X)) ) * 100;

    % Find number of PCs for which explained variance exceeds OrigVar
    k = length(find(explained >= OrigVar));

else %use classic Kaiser Rule based on eigenvalues
    C = cov(X);
    eigenvalues = eig(C);

    % Find number of PCs for which eigenvalue exceeds 1
    k = length(find(eigenvalues >= 1));
end
end

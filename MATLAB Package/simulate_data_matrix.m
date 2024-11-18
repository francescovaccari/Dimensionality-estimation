function varargout = simulate_data_matrix(fake_units, time_series_length, num_latent, gen, DF, NF, equal_noise, soft_norm, display)

% SIMULATE_DATA_MATRIX: Simulates a data matrix based on the linear combinations of
% subset of latent variables plus noise. Several parameters can be tuned for 
% simulating various data types (e.g., neural data). 
%
% Inputs:
%   fake_units        - Number of variables (e.g., synthetic neurons) in the generated data.
%   time_series_length - Length of the time series (number of time points).
%   num_latent        - Number of latent variables that will define the 
%                       underlying structure of the data.
%   gen               - String specifying the latent variable generation method:
%                        'random'    - Generate stochastic random time series.
%                        'sine'      - Generate sinusoidal time series.
%                        'PCs'       - Generate time series based on real data dynamics.
%   DF                - Decay factor for scaling latent variables. It is
%                       the exponential of exponential decay curve, thus
%                       values should be in the range [0 0.5] depending also on
%                       num_latent and noise (namely, NF value, see below)
%   NF                - Noise factor to scale the noise added to the data.
%   equal_noise       - Boolean to specify whether the noise is distributed equally 
%                       across all fake units (neurons) or not.
%   soft_norm         - Boolean to apply soft normalization (as opposed to standard z-score normalization).
%   display           - Boolean to control whether plots of the generated data should be shown.
%
% Outputs:
%   X                 - The simulated neural data matrix with 'fake_units' units (rows) 
%                       and 'time_series_length' time points (columns), including noise and scaling.
%   U                 - The latent variable matrix (before noise is added).
%   noise_matrix   - The simulated data matrix without noise.
%   L                 - The latent variables (before orthogonalization and scaling).
%
% Example:
%   [X, L_var, L, U, noise_matrix] = simulate_data_matrix(100, 1000, 5, 'sine', 0.1, 1, false, true, true);
%   
% Reference:
%   The simulation procedure is described in detail in:
%
%
% Author: [Francesco E. Vaccari, PhD]
% Date: [12/11/2024]

switch gen
    case 'random'
        % Generate random time series (stochastic)
        L = generate_random_time_series(num_latent, time_series_length);
    case 'sine'
        % Generate sine-wave based time series
        L = generate_random_sine_series(num_latent, time_series_length);
    case 'PCs'
        % Generate time series based on principal components
        L = generate_random_units_from_PCs(num_latent, time_series_length);
end

% Pre-process latent data (z-score, orthogonalize, and z-score again)
L = zscore(L);

if display
    % Plot statistics of latent variables BEFORE orthogonalization and scaling
    figure
    subplot(1, 3, 1)
    bar(var(L))
    ylabel('Variance')
    xlabel('Dimensions')
    title('Latent Variables Variance')
    
    subplot(1, 3, 2)
    imagesc(corr(L))
    ylabel('Dimensions')
    xlabel('Dimensions')
    title('Latent Variables Correlation')
    
    subplot(1, 3, 3)
    for l = 1:size(L, 2)
        plot(L(:, l) + ones(size(L, 1), 1) * 2 * l)
        hold on
    end
    title('Latent Variables Temporal Evolution')
    sgtitle('Generated Real Latent Variables Statistics')
end

% Orthogonalize latent variables using PCA (Principal Component Analysis)
[~, L, ~] = pca(L, 'NumComponents', num_latent);
L = zscore(L); % Z-score after PCA

% Apply decay factor to scale latent variables
SF = exp(-DF * [1:num_latent]);
SF = SF / sum(SF); % Normalize scaling factors
for i = 1:num_latent
    L(:, i) = L(:, i) * SF(i);
end

if display
    % Plot statistics of latent variables AFTER orthogonalization and scaling
    figure
    subplot(1, 3, 1)
    bar(var(L))
    ylabel('Variance')
    xlabel('Dimensions')
    title('Latent Variables Variance')
    
    subplot(1, 3, 2)
    imagesc(corr(L))
    ylabel('Dimensions')
    xlabel('Dimensions')
    title('Latent Variables Correlation')
    
    subplot(1, 3, 3)
    for l = 1:size(L, 2)
        plot(L(:, l) + ones(size(L, 1), 1) * l)
        hold on
    end
    title('Latent Variables Temporal Evolution (After Scaling)')
    sgtitle('Generated Real Latent Variables Statistics AFTER Scaling')
end

% Generate random weight matrix W
W = rand(num_latent, fake_units) - 0.5;

% Normalize the weight matrix
W = W ./ sqrt(sum(W.^2));  % Normalize the columns to have unit norm

% Create 'U' matrix by combining real_latents with W
U = L * W;
U = zscore(U);  % Z-score 'U' matrix


% Generate noise matrix
noise_matrix = normrnd(0, 1, size(U));
noise_matrix = zscore(noise_matrix);  % Z-score the noise matrix

% Adjust noise factor
NF = NF * mean(std(U));  % Normalize the noise matrix
noise_matrix = NF * noise_matrix;

% Apply different noise distribution depending on equal_noise flag
if equal_noise
    beta = ones(1, fake_units); % Equal noise distribution across all units
else
    beta = normrnd(1, 1/3, 1, fake_units); 
    beta(beta < 0 | beta > 2) = 0;  % Clip values to [0, 2]
end

% Add noise to the data matrix X
X = U .* (2 - beta) + noise_matrix .* beta;  % Weighted noise addition
X = X + abs(min(X));  % Ensure that all values in X are non-negative (firing rate)

% Apply normalization if specified
if soft_norm
    X = soft_normalize(X, 2, mean(X, 'all'));
    X = X - mean(X);  % Center the data
else
    X = zscore(X);  % Standard z-score normalization
end

if display
    % Visualize the final simulated data matrix
    figure
    subplot(1, 2, 1)
    imagesc(X)
    colormap('hot')
    colorbar
    caxis([-3 3])
    
    subplot(1, 2, 2)
    for l = 1:3
        plot(X(:, randi(fake_units, 1, 1)) + ones(size(L, 1), 1) * 3 * l, 'LineWidth', 2)
        hold on
    end
    sgtitle('Simulated Fake Units')
end

% Estimate the latent variables' contribution to the data matrix X
b = L(:, :) \ X;  % Solve the linear system X = real_latents * b

% Calculate variance explained by each latent variable
L_var = [];
for i = 1:num_latent
    L_var(i) = 100 - (var(X - L(:, i) * b(i, :)) / var(X)) * 100;
end
[L_var, I] = sort(L_var, 'descend');  % Sort by variance explained

% Perform PCA on the final data matrix X
[coeff, last_score, latent, tsquared, score_var, mu] = pca(X); 
score_var = score_var';  % Variance explained by each principal component

if display
    % Plot the variance explained by real latent variables vs PCs
    figure
    bar([L_var zeros(1, fake_units - num_latent); score_var]')
    ylabel('Explained Variance')
    hold on
    yyaxis right
    plot(cumsum(L_var))  % Cumulative explained variance for real latent variables
    plot(cumsum(score_var))  % Cumulative explained variance for PCs
    legend('Real Latent Variance', 'PCs Variance', 'Real Latent Variance Cumsum', 'PCs Variance Cumsum')
    xlabel('Dimensions')
    ylabel('Total Explained Variance')
end

% Output the results
varargout{1} = X; % Always return X as the first output
if nargout > 1
    varargout{2} = L_var; % Return variance explained by the latent variables used to generate the data if requested
end
if nargout > 2
    varargout{3} = L; % Return L if requested
end
if nargout > 3
    varargout{4} = U; % Return U if requested
end
if nargout > 4
    varargout{5} = noise_matrix; % Return noise_matrix if requested
end

end

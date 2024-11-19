clear all; close all; clc;

%% Script Overview:
% This script generates a synthetic data matrix based on various parameters 
% such as the number of units, time series length, and latent variables.
% It then estimates the dimensionality of the data using different methods 
% (e.g., variance threshold, participation ratio, Kaiser rule, parallel 
% analysis, and cross-validation). The results are plotted in a bar chart 
% with a label indicating the true dimensionality.

%% 1. Generate Synthetic Data Matrix
% This section defines the parameters for generating synthetic data.
% The function `simulate_data_matrix` is called to create a data matrix
% that will serve as the input for the dimensionality estimation methods.

% Define generation parameters for synthetic data
fake_units = 50;            % Number of variables (e.g., synthetic neurons) in the generated data.
time_series_length = 500;   % Length of the time series (number of time points).
num_latent = 6;            % Number of latent variables that will define the underlying structure of the data.
gen = 'PCs';               % Latent variable generation method (specifying PCs here).
DF = 0.15;                  % Decay factor for scaling latent variables. Range: [0, 0.5].
NF = 1;                    % Noise factor to scale the noise added to the data.
equal_noise = false;       % Boolean to specify whether the noise is equally distributed across all fake units.
soft_norm = true;          % Boolean to apply soft normalization (soft z-score normalization).
display = true;            % Boolean to control whether plots of the generated data are shown.

% Generate synthetic data using the above parameters
X = simulate_data_matrix(fake_units, time_series_length, num_latent, gen, DF, NF, equal_noise, soft_norm, display);

X = X - mean(X); %mean centering

% Calculating the a posteriori tau (i.e., the exponential of the variance explained
% decrease) on the generated data
[tau, coefficients] = fit_tau(X, 1);

%% 2. Dimensionality Estimation Using Different Methods
% In this section, various methods for estimating the dimensionality (number 
% of principal components) of the data are applied. Each method provides 
% a different estimate, which is used to compare and assess the underlying 
% dimensionality.

% Apply the variance hard threshold method (90% explained variance)
k_90 = variance_hard_threshold(X, 90);  % Estimate the dimensionality for 90% explained variance

% Apply the Participation Ratio (PR) method
k_PR = participatio_ratio(X);  % Estimate the dimensionality using the Participation Ratio

% Apply the Kaiser rule method (based on eigenvalue variance)
k_K1 = kaiser_rule(X, 'UseVariance', true);  % Estimate the dimensionality with the Kaiser Rule

% Apply the Parallel Analysis (PA) method
k_PA = parallel_analysis(X, 0.95, 1000);  % Estimate the dimensionality with Parallel Analysis (95% quantile, 1000 replications)

% Apply Cross-Validated PCA (CV) method (5-fold, 15 max dimensions, 10^-6 tolerance, 100 max iterations)
k_CV = eval_num_PCs_cross_val(X, 5, 15, 10^-4, 100, true);  % Estimate the dimensionality using cross-validation (PCA)

%% 3. Plot the Dimensionality Estimates
% This section creates a bar chart to visualize the results of all dimensionality 
% estimation methods, including a horizontal line that indicates the true 
% dimensionality. The true dimensionality is labeled on the plot.

% Create a figure for the bar chart
figure

% Create the bar plot showing dimensionality estimates for each method
bar([k_90 k_PR k_K1 k_PA k_CV])
hold on

% Add a horizontal line indicating the 'True' dimensionality
yline(num_latent, 'k', 'LineWidth', 1.5);  % Add a black horizontal line at the true dimensionality

% Label the y-axis
ylabel('Estimated dimensionality')

% Add a text label near the true dimensionality line
text(max(xlim) * 0.80, num_latent, 'True dimensionality', ...   % Place text near the true dimensionality line
    'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left', ...
    'FontSize', 10, 'Color', 'k')  % Set label properties (font size, color)

% Set custom x-axis tick labels
xticklabels({'90% thres','PR','K1','PA','CV'})  % Label each bar with the corresponding method name

% Add a title to the plot
title('Dimensionality Estimation Results')


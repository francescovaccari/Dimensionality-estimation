clear all; close all; clc;

%% Script Overview:
% This script applies different dimensionality estimation criteria on
% synthetic data. It compares various approaches including variance threshold,
% participation ratio, Kaiser criterion, parallel analysis, and cross-validation.

%% 1. Generate Synthetic Data Matrix
% Parameters for synthetic data generation
params = struct();
n_units = 50;                % Number of variables (neurons)
n_timepoints = 500;          % Length of time series
n_latent = 6;                % True dimensionality
params.gen = 'dynamical'; % Method for generating latent variables
params.decayFactor = 0.15;         % Decay factor for scaling / normally in the range [0, 0.5]
params.noiseFactor = 1;            % Scale factor for noise (0 for no noise, 1 for 50% of the variance explained by noise)
param.noiseDistr = 'gaussian';    % Type of noise distribution: 'gaussian' or 'poisson'
params.equalNoise = false;         % Whether noise is equally distributed
params.soft_norm = true;       % Apply soft normalization (true) or zscoring (false)
params.display = true;           % Display diagnostic plots

% Generate synthetic dataset
X = simulate_data_matrix(n_units, n_timepoints, n_latent, params);

% Center the data
X = X - mean(X, 2);

% Calculate decay constant tau

[tau, coeffs] = fit_tau(X, params.display);

%% 2. Dimensionality Estimation

% Initialize dimensionality estimation parameters

variance_threshold = 90;  % For variance threshold method
pa_confidence = 95;       % Confidence level for parallel analysis
pa_iterations = 1000;     % Number of PA iterations
CV_max_dims = 15;        % Maximum dimensions to test in CV
CV_params = struct();
CV_params.type = 'rows&columns_imputation'; % type of CV to be performed
CV_params.cv_folds = 5;            % Number of cross-validation folds
CV_params.cv_tolerance = 1e-3;      % Convergence tolerance for CV
CV_params.cv_max_iter = 50;       % Maximum iterations for CV
CV_params.display = true;

% Estimate dimensionality using different methods
estimates = struct();
estimates.var_thresh = variance_hard_threshold(X, variance_threshold);
estimates.pr = participation_ratio(X);
estimates.kaiser = kaiser_rule(X, 'UseVariance', true);
estimates.pa = parallel_analysis(X, pa_confidence/100, pa_iterations);
estimates.cv = eval_num_PCs_cross_val(X, CV_max_dims, CV_params);

%% 3. Visualization
figure('Name', 'Dimensionality Estimation Comparison');

% Create bar plot
methods = {'variance thresh', 'PR', 'K1', 'PA', 'CV'};
dims = [estimates.var_thresh, estimates.pr, estimates.kaiser, estimates.pa, estimates.cv];
b = bar(dims);
hold on

% Add true dimensionality line
yline(n_latent, 'k', 'LineWidth', 1.5);

% Customize plot
ylabel('Estimated Dimensionality')
set(gca, 'XTickLabel', methods)
title('Comparison of Dimensionality Estimation Methods')

% Add true dimensionality label
text(length(methods) * 0.8, n_latent, 'True dimensionality', ...
    'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left', ...
    'FontSize', 10, 'Color', 'k')

% Add grid and adjust appearance
grid on
box on
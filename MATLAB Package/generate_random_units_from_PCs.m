function real_latents = generate_random_units_from_PCs(num_latent, time_series_length)
% GENERATE_RANDOM_UNITS_FROM_PCS: Generates random latent variables based on 
% real data principal components (PCs). This function loads a precomputed set of 
% principal component (PC) scores from a data file, selects a subset of PCs, 
% and extracts a time series based on these selected components.
%
% Inputs:
%   num_latent        - The number of latent variables (PCs) to extract from the real data.
%   time_series_length - The number of time points (rows) for the generated time series.
%
% Outputs:
%   real_latents      - A matrix of size (time_series_length x num_latent) where each column 
%                       represents a time series generated from a different principal component.
%
% Example:
%   real_latents = generate_random_units_from_PCs(10, 500);
%   % Generates a 500x10 matrix of latent variables based on 10 principal components.

% Author: Francesco E. Vaccari, PhD
% Date: 12/11/2024

% Load the principal components (PCs) from the example data file
% This assumes that the 'example_time_series.mat' file contains the 'score' variable,
% which is the matrix of principal component scores.
load('example_time_series.mat')

% Randomly select 'num_latent' principal components from the available set of PCs
pcX = randperm(size(score, 2), num_latent);

% Extract the first 'time_series_length' time points from the selected PCs
real_latents = score(1:time_series_length, pcX);

end

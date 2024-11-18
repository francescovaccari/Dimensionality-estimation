function real_latents = generate_random_sine_series(num_latent, time_series_length)
% GENERATE_RANDOM_SINE_SERIES: Generates a matrix of sine waves with different frequencies.
% Each column in the output matrix represents a different latent variable with a sinusoidal signal.
% The frequency of each sine wave increases exponentially with the index of the latent variable.
%
% Inputs:
%   num_latent        - The number of latent variables (columns) to generate.
%   time_series_length - The number of time points (rows) for each sine wave (i.e., the length of the time series).
%
% Outputs:
%   real_latents      - A matrix of size (time_series_length x num_latent), where each column contains
%                       a sine wave with a different frequency. The waves are generated randomly with
%                       respect to phase (random phase shift added).
%
% Example:
%   real_latents = generate_random_sine_series(5, 1000);
%   % Generates a 1000x5 matrix with 5 sine waves of increasing frequencies.

% Author: [Francesco E. Vaccari, PhD]
% Date: [12/11/2024]

% Define the frequencies for the sine waves
% Frequencies increase exponentially with index (e.g., 2^1, 2^2, ..., 2^num_latent)
for i = 1:num_latent
    frequencies(i) = 2^i;  % Frequency for the i-th latent variable
end

% Define the time vector (from 0 to 2*pi, with 'time_series_length' points)
t = linspace(0, 2, time_series_length) * pi;  % Scale time to [0, 2*pi]

% Initialize the matrix for the latent variables
real_latents = zeros(time_series_length, num_latent);

% Fill the 'real_latents' matrix with sinusoidal waves of different frequencies
for i = 1:num_latent
    % For each latent variable, generate a sine wave with a random phase shift
    real_latents(:, i) = sin(frequencies(i) * t + rand(1, 1));  % Add random phase shift
end

end

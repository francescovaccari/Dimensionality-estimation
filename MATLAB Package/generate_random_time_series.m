function real_latents = generate_random_time_series(num_latent, time_series_length)
% GENERATE_RANDOM_TIME_SERIES: Generates a matrix of random time series where each time series is 
% modeled as a random walk, with each time point depending on the previous
% one plus a random value. Each column in the output matrix represents a different latent variable.
%
% Inputs:
%   num_latent        - The number of latent variables (columns) to generate.
%   time_series_length - The number of time points (rows) for each time series (i.e., the length of the time series).
%
% Outputs:
%   real_latents      - A matrix of size (time_series_length x num_latent), where each column represents
%                       a random walk time series. The value at each time point is the sum of the previous value
%                       and a normally distributed random increment.
%
% Example:
%   real_latents = generate_random_time_series(5, 1000);
%   % Generates a 1000x5 matrix of random walk time series with 5 latent variables.

% Author: [Francesco E. Vaccari, PhD]
% Date: [12/11/2024]

% Initialize the matrix to hold the random time series
real_latents = zeros(time_series_length, num_latent);

% Loop over the latent variables (columns)
for i = 1:num_latent
    % Generate a random walk for each latent variable
    for t = 1:time_series_length
        if t == 1
            % For the first time point, sample a random value from a normal distribution (mean=0, std=1)
            real_latents(t, i) = normrnd(0, 1, 1, 1); 
        else
            % For subsequent time points, add a random increment (normal distribution) to the previous value
            real_latents(t, i) = real_latents(t-1, i) + normrnd(0, 1, 1, 1);
        end
    end
end

end

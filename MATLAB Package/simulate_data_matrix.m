function [X, varargout] = simulate_data_matrix(fake_units, time_series_length, num_latent, settings)
%--------------------------------------------------------------------------
% SIMULATE_DATA_MATRIX Generate synthetic data with controlled latent structure
%
% Designed for generating synthetic neural-like data with known underlying dimensionality and 
% noise characteristics. Simulates a data matrix based on linear combinations of latent variables plus
% noise. The degree of non-linearity mapping between latent activations and simulated neural activity
% can be tuned. Other parameters can be varied (see below).
%
% Syntax:
%   X = simulate_data_matrix(fake_units, time_series_length, num_latent)
%   [X, output] = simulate_data_matrix(fake_units, time_series_length, num_latent, settings)
%   [X, output, settings] = simulate_data_matrix(fake_units, time_series_length, num_latent, settings)
%
% Inputs:
%   fake_units          - Number of variables/units to generate (e.g., neurons)
%   time_series_length  - Length of time series (number of timepoints)
%   num_latent          - Number of latent variables defining data structure
%   settings           - (Optional) struct with fields:
%       .gen           - Latent generation method: 'random','sine','PCs','dynamical' (default:'random')
%       .noiseDistr    - Noise distribution: 'gaussian' or 'poisson' (default:'gaussian')
%       .nonLinAlfa    - Non-linearity strength parameter (default:0)
%       .decayFactor   - Decay factor for scaling latents (default:0.5)
%       .decayCap      - Cap decay to maintain latent structure (default:false)
%       .noiseFactor   - Scale factor for noise amplitude (default:1)
%       .equalNoise    - Use equal noise across units (default:false)
%       .soft_norm     - Use soft normalization instead of z-score (default:false)
%       .display       - Plot diagnostic figures (default:false)
%
% Outputs:
%   X        - Simulated data matrix (time_series_length x fake_units)
%   output   - (Optional) struct with fields:
%       .single_L_var    - Variance explained by each latent
%       .single_L_SNR    - Signal-to-noise ratio for each latent
%       .L_R2s          - R-squared values for latent reconstructions
%       .L_var          - Total variance explained by linear structure
%       .U_var          - Total variance in combined latents
%       .noise_var      - Variance explained by noise
%       .nonLinear_var  - Variance from non-linear effects
%       .L              - Raw latent variables matrix
%       .U              - Combined latent representation
%       .noise_matrix   - Generated noise matrix
%       .tau            - variance decay for X
%       .settings       - Copy of input settings
%       .fake_units     - Number of units generated
%       .time_series_length - Length of time series
%       .num_latent     - Number of latent dimensions
%   settings - (Optional) Copy of settings struct with defaults filled in
%
% Example:
%   % Generate 100 units with 5 latents over 1000 timepoints
%   settings = struct('gen','sine','noiseFactor',0.5,'display',true);
%   [X,stats] = simulate_data_matrix(100, 1000, 5, settings);
%
% Author: Francesco E. Vaccari, PhD
% Date: September 24, 2025
%--------------------------------------------------------------------------

% --- Handle settings structure input ---
if nargin < 4
    settings = struct();
end
if ~isfield(settings, 'gen') | isnan(settings.gen),                   settings.gen = 'random'; end
if ~isfield(settings, 'noiseDistr') | isnan(settings.noiseDistr),            settings.noiseDistr = 'gaussian'; end
if ~isfield(settings, 'nonLinAlfa') | isnan(settings.nonLinAlfa),            settings.nonLinAlfa = 0; end
if ~isfield(settings, 'decayFactor') | isnan(settings.decayFactor),           settings.decayFactor = 0.5; end
if ~isfield(settings, 'decayCap') | isnan(settings.decayCap),               settings.decayCap = false; end
if ~isfield(settings, 'noiseFactor') | isnan(settings.noiseFactor),           settings.noiseFactor = 1; end
if ~isfield(settings, 'equalNoise') | isnan(settings.equalNoise),            settings.equalNoise = false; end
if ~isfield(settings, 'soft_norm') | isnan(settings.soft_norm),             settings.soft_norm = false; end
if ~isfield(settings, 'display') | isnan(settings.display),               settings.display = false; end

% Simulate!
if settings.decayCap %if decay is capped

    settingsTemp = settings;
    settingsTemp.nonLinAlfa = 0; %set linear simulation
    settingsTemp.display = false;

    while true
        [X, output] = simulate(fake_units, time_series_length, num_latent, settingsTemp);
%         [coeff, score] = pca(X);
% 
%         for k = 1:num_latent
%             Xhat =  score(:,1:k)*coeff(:,1:k)';
%             error(k) = sumsqr(output.U - Xhat);
%         end

        [k_opt, vv] = find_optimality(X, output);

        %         if error(num_latent) > min(error) %any(output.single_L_SNR < 2/3) %if the last PC contains more noise than signal
        if k_opt ~= num_latent
            settingsTemp.decayFactor = settingsTemp.decayFactor*0.8; %reduce the decayFactor by 10% and run another simulation
        else
            settings.decayFactor = settingsTemp.decayFactor; %set the final decayFactor to be equal to the last one
            break
        end

    end

else
    [X, output] = simulate(fake_units, time_series_length, num_latent, settings); %run the output simulation
end



% Check the number of outputs requested and return accordingly
if nargout == 2
    varargout{1} = output;  % Return the struct as the second output
elseif nargout == 3
    varargout{1} = output;  % Return the struct as the second output
    varargout{2} = settings;  % Return the struct as the third output
end


    function [X, output] = simulate(fake_units, time_series_length, num_latent, settings)

        % Assign settings fields to variables used below without changing any other commands
        gen         = settings.gen;
        decayFactor          = settings.decayFactor;
        decayCap = settings.decayCap;
        noiseFactor          = settings.noiseFactor;
        equalNoise = settings.equalNoise;
        soft_norm   = settings.soft_norm;
        display     = settings.display;

        tmp_num_latent = num_latent;
        for i = 1:10
            switch gen
                case 'random'
                    % Generate random time series (stochastic)
                    L = generate_random_time_series(tmp_num_latent, time_series_length);
                case 'sine'
                    % Generate sine-wave based time series
                    L = generate_random_sine_series(tmp_num_latent, time_series_length);
                case 'PCs'
                    % Generate time series based on principal components
                    L = generate_random_units_from_PCs(tmp_num_latent, time_series_length);
                case 'dynamical'
                    L = generate_random_dynamical_series(tmp_num_latent, time_series_length);
            end


            % Pre-process latent data (z-score, orthogonalize, and z-score again)
            L = zscore(L);

            if rank(L) == num_latent
                break
            else
                tmp_num_latent = tmp_num_latent+1;
            end

        end

        num_latent = rank(L);

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
        SF = exp(-decayFactor * [1:num_latent]); % following a exponential decay
        % SF = [1:num_latent] .^ (-decayFactor); % following a power law function

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
        W = normrnd(0, 1, num_latent, fake_units); %Draw samples from independent Guassian distributions to avoid directional biases

        % Normalize the weight matrix
        W = W ./ sqrt(sum(W.^2));  % Normalize the columns to have unit norm

        % Create 'U' matrix by combining real_latents with W
        U = L * W;

        % Add non-linearities if required
        if settings.nonLinAlfa > 0
            U = normalize(U,"range", [0 1]); % To be comparable with Altan et al. (2021) procedure
            U = (exp(settings.nonLinAlfa*U)-1) / (exp(settings.nonLinAlfa)-1);
        end

        % Generate noise matrix
        switch settings.noiseDistr
            case 'poisson'
                U = normalize(U,"range", [0 5]); %assuming a 50ms bin, it would correspond to a maximum FR around 100 sp/sec
                noise_matrix = poissrnd(U) - U; % Generate Poisson noise where each element's variance equals its value in U (subtract U since each generated element is expected to be equal to U)

            case 'gaussian'
                noise_matrix = normrnd(0, 1, size(U));
        end

        U = zscore(U);  % Z-score 'U' matrix
        noise_matrix = zscore(noise_matrix);  % Z-score the noise matrix

        % Adjust noise factor
        noiseFactor = noiseFactor * mean(std(U)./std(noise_matrix));  % Equilibrate noise and signal variances if not already equilibrated
        noise_matrix = noiseFactor * noise_matrix;

        % Apply different noise distribution depending on equalNoise flag
        if equalNoise
            beta = ones(1, fake_units)*0.5; % Equal noise distribution across all units
        else
            beta = normrnd(0.5, 1/6, 1, fake_units);
            beta(beta < 0) = 0; beta(beta > 1) = 1;  % Clip values to [0, 1]
        end

        % Add noise to the data matrix X
        X = U .* (1 - beta) + noise_matrix .* beta;  % Weighted noise addition
        X = X + abs(min(X));  % Ensure that all values in X are non-negative (firing rate)

        % Apply normalization if specified
        if soft_norm
            X = soft_normalize(X, 2, mean(X, 'all')/2);
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
        b = L \ X;  % Solve the linear system X = real_latents * b

        % Calculate variance explained by each latent variable
        single_L_var = [];
        for i = 1:num_latent
            single_L_var(i) = 100 - (var(X - L(:, i) * b(i, :)) / var(X)) * 100;
        end
        [single_L_var, I] = sort(single_L_var, 'descend');  % Sort by variance explained

        L_R2s = [];
        for i = 1:num_latent
            L_R2s(i) = Rsquared(X, L(:, 1:i) * b(1:i, :));
        end

        L_var = 100 - (var(X - L * b) / var(X)) * 100;

        % Calculate how much the dataset is non-linear
        Xhat = [];
        for j = 1:fake_units
            b =  U(:,j) \ X(:,j);
            Xhat = [Xhat, U(:,j)*b];
        end
        U_var = 100 - var(X - Xhat)/var(X)*100;

        nonLinear_var = U_var - L_var;

        % Calculate how much of the dataset is noise
        Xhat = [];
        for j = 1:fake_units
            b =  noise_matrix(:,j) \ X(:,j);
            Xhat = [Xhat, noise_matrix(:,j)*b];
        end
        noise_var = 100 - var(X - Xhat)/var(X)*100; % Calculate the real noise variance percentage

        % calculate tau on X
        [tau, ~] = fit_tau(X, false);

        % Perform PCA on the final data matrix X
        [coeff, last_score, latent, tsquared, score_var, mu] = pca(X);
        score_var = score_var';  % Variance explained by each principal component
        if numel(score_var) < fake_units
            score_var = [score_var zeros(1, fake_units-numel(score_var))];
        end

        if display
            % Plot the variance explained by real latent variables vs PCs
            figure
            bar([single_L_var, zeros(1, fake_units - num_latent); score_var]')
            ylabel('Explained Variance')
            hold on
            yyaxis right
            plot(cumsum(single_L_var))  % Cumulative explained variance for real latent variables
            plot(cumsum(score_var))  % Cumulative explained variance for PCs
            legend('True Latent Variance', 'PCs Variance', 'True Latent Variance Cumsum', 'PCs Variance Cumsum')
            xlabel('Dimensions')
            ylabel('Total Explained Variance')
            title(['Data underlying linear structure / linearities = ' num2str(L_var) ...
                ' + nonlinearities = ' num2str(nonLinear_var) ...
                ' + noise = ' num2str(noise_var) '% of the X variance'])
        end

        % --- Return the outputs in a struct ---
        output.single_L_var = single_L_var;
        jj = min([num_latent fake_units]);
        output.single_L_SNR = single_L_var(1:jj)./score_var(1:jj);
        output.L_R2s        = [L_R2s];
        output.L_var        = L_var;
        output.U_var        = U_var;
        output.noise_var    = noise_var;
        output.nonLinear_var= nonLinear_var;
        output.L            = L;
        output.U            = U;
        output.noise_matrix = noise_matrix;
        output.tau          = tau;
        output.settings     = settings;
        output.fake_units   = fake_units;
        output.time_series_length = time_series_length;
        output.num_latent   = num_latent;

    end
end

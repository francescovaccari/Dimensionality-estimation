function [k, varargout] = eval_num_PCs_cross_val(X, n_cross, n_dims_max, tol, max_iter, display)

% eval_num_PCs_cross_val performs cross-validation to choose the
% optimal number of Principal Components (PCs) using R2 as the performance metric.
% The function compares the reconstruction error and reports the best number of PCs
% based on average R2 across cross-validation folds. Cross-validation
% aims to remove test elements along rows AND columns at the same time.
% The algorithm is based on iterations to update the PCA model and
% the estimates for the test elements alternatively until convergence.
%
% INPUT:
%   X           - Data matrix (rows are observations, columns are variables)
%   n_cross     - Number of cross-validation folds
%   n_dims_max  - Maximum number of PCs to evaluate
%   tol         - Tolerance for convergence (relative error)
%   max_iter    - Maximum number of iterations for convergence
%   display     - Flag (0 or 1) to display results visually
%
% OUTPUT:
%   crossval_PCA - Struct containing:
%       - avg_R2: Average R2 scores across all folds
%       - std_R2: Standard deviation of R2 scores
%       - SEM_R2: Standard error of the mean of R2 scores
%       - R2_best: Best R2 score
%       - idx_best: Index of the best number of PCs
%       - R2_1SEfrombest: R2 score corresponding to 1 standard error from best
%       - idx_1SEfrombest: Index of the number of PCs corresponding to 1 SE from best
%       - R2_null: R2 scores for a null model
%
% Example usage:
%   crossval_PCA = eval_num_PCs_cross_val(X, 10, 25, 10^-4, 100, 0);
%
%   This would return the output structure containing results given the
%   matrix X, performing 10-folds CV, maximum number of dimensions evaluate equal to
%   25, tolerance for convergnce equal to 10^-4, maximum iterations equal
%   to 100. No graphical display is generated (display flag set to 0).
%
% Author: [Francesco E. Vaccari, PhD]
% Date: [12/11/2024]

X = X - mean(X); %just to be sure, since it can disrupt results

% Initialize cross-validation partition
c = cvpartition(numel(X), 'KFold', n_cross);
R2 = [];  % Store R2 values for each fold and number of PCs
R2_null = [];  % Store R2 values for null model

% Loop over cross-validation folds
for cross = 1:n_cross
    idx_test = test(c, cross);  % Test set indices for this fold

    % Loop over different numbers of PCs (k)
    for k = 1:n_dims_max
        Z = X;
        Z(idx_test) = 0;  % Set test set values to zero
        Z_past = rand(size(Z));  % Initialize random past values
        err = norm(Z - Z_past) / norm(Z_past);  % Initial error
        iter = 1;  % Initialize iteration counter

        % Perform PCA until convergence or maximum iterations
        while err > tol && iter < max_iter
            [coeff, score, latent] = pca(Z, "NumComponents", k);  % Compute k PCs
            Z_new = score * coeff';  % Reconstruct data with k PCs
            Z(idx_test) = Z_new(idx_test);  % Update the test set values
            err = norm(Z(idx_test) - Z_past(idx_test)) / norm(Z_past(idx_test));  % Compute error
            Z_past = Z;  % Update past values for next iteration
            iter = iter + 1;

            % If max iterations reached, display warning
            if iter > max_iter
                disp('Stopping for reaching max iterations allowed');
            end
        end

        % Calculate R2 for PCA reconstruction error
        real_values = X(idx_test);  % True values for test set
        predicted_values = Z(idx_test);  % Predicted values for test set
        SST = sum((real_values - mean(real_values)).^2);  % Total sum of squares
        SSE = sum((real_values - predicted_values).^2);  % Sum of squares error
        R2(cross, k) = 1 - SSE / SST;  % R2 score for this fold and k PCs

        % Null model (randomly generated orthogonal components)
        d = size(X, 2);
        v = rand(d, 1);
        vbar = v.' / norm(v);  % Normalize v
        Q = null(vbar).';  % Orthogonal complement of v
        orthoX = [vbar; Q];  % Create orthonormal basis

        % Null model reconstruction and R2 calculation
        score_null = X * orthoX(:, 1:k);
        X_rec = score_null * orthoX(:, 1:k)';  % Reconstruct with null model
        real_values = X(idx_test);
        predicted_values = X_rec(idx_test);
        SST = sum((real_values - mean(real_values)).^2);
        SSE = sum((real_values - predicted_values).^2);
        R2_null(cross, k) = 1 - SSE / SST;  % R2 score for null model

        % Display current fold and k
        disp(['Fold ', num2str(cross), ', k = ', num2str(k)]);
    end
end

% Compute average, standard deviation, and SEM for R2 scores
avg_R2 = mean(R2);  % Average R2 across folds
std_R2 = std(R2);  % Standard deviation of R2
SEM_R2 = std_R2 / sqrt(n_cross);  % Standard error of the mean

% Find the best R2 score and its corresponding k (number of PCs)
[R2_best, idx_best] = max(avg_R2);

% Find the index of the number of PCs where R2 is within one standard error from the best
idx_1SEfrombest = min(find(avg_R2 >= (R2_best - SEM_R2(idx_best))));

% R2 value corresponding to 1 SE from the best R2 score
R2_1SEfrombest = avg_R2(idx_1SEfrombest);

% Display results if requested
if display == 1
    figure;
    errorbar(avg_R2, SEM_R2);  % Plot average R2 with error bars
    hold on;
    yline(R2_best, 'k');  % Line at best R2
    xline(idx_best, 'k');  % Line at best k
    yline(R2_1SEfrombest, 'r');  % Line at R2 corresponding to 1 SE from best
    xline(idx_1SEfrombest, 'r');  % Line at 1 SE from best k
    legend({'avg cross-validated R2 ± SEM', 'Best', '1 SE from best'});
    ylabel('crossvalidate R2')
    xlabel('Dimensions')
    title('Cross-validation results')
end

% Prepare output structure
crossval_PCA.avg_R2 = avg_R2;
crossval_PCA.std_R2 = std_R2;
crossval_PCA.SEM_R2 = SEM_R2;
crossval_PCA.R2_best = R2_best;
crossval_PCA.idx_best = idx_best;
crossval_PCA.R2_1SEfrombest = R2_1SEfrombest;
crossval_PCA.idx_1SEfrombest = idx_1SEfrombest;
crossval_PCA.R2_null = R2_null;

k = idx_best;

% Check the number of outputs requested and return accordingly
if nargout == 1
    varargout{1} = crossval_PCA;  % Return the crossval_PCA struct as the second output 
end

end

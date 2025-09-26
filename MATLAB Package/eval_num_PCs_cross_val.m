function [k, varargout] = eval_num_PCs_cross_val(X, n_dims_max, settings)
%--------------------------------------------------------------------------
%
% Performs cross-validation to select the optimal number of Principal Components (PCs)
% for PCA reconstruction, using R2 as the performance metric to be maximized. Supports multiple
% cross-validation schemes (rows, columns, rows&columns, bi-cross) and iterative
% imputation for missing data. Returns the best number of PCs and detailed statistics.
%
% Syntax:
%   k = eval_num_PCs_cross_val(X, n_dims_max)
%   [k, crossval_PCA] = eval_num_PCs_cross_val(X, n_dims_max, settings)
%
% Inputs:
%   X           - Data matrix (observations x variables)
%   n_dims_max  - Maximum number of PCs to evaluate (positive integer)
%   settings    - (Optional) struct with fields:
%                   .type      : 'rows', 'columns', 'rows&columns_imputation',
%                                'rows&columns_ALS' (can be very slow!), or 'bi-cross' (default: 'rows&columns_imputation')
%                   .n_cross   : Number of cross-validation folds (default: 5)
%                   .tol       : Convergence tolerance for iterative imputation (default: 1e-3)
%                   .max_iter  : Maximum iterations for imputation (default: 100)
%                   .display   : 0/1, plot results if 1 (default: 0)
%
% Outputs:
%   k               - Optimal number of PCs (integer)
%   crossval_PCA    - (Optional) struct with fields:
%                       .avg_R2             : Mean R2 across folds for each PC count
%                       .std_R2             : Standard deviation of R2
%                       .SEM_R2             : Standard error of mean R2
%                       .R2_best            : Best mean R2 value
%                       .idx_best           : Index of best PC count
%                       .R2_1SEfrombest     : R2 at 1 SEM from best
%                       .idx_1SEfrombest    : Index at 1 SEM from best
%                       .R2_1STDfrombest    : R2 at 1 STD from best
%                       .idx_1STDfrombest   : Index at 1 STD from best
%                       .R2_null            : R2 scores for null model
%
% Example:
%   [k, stats] = eval_num_PCs_cross_val(X, 25, struct('type','rows','n_cross',10,'tol',1e-4,'max_iter',100,'display',1));
%
% Author: Francesco E. Vaccari, PhD
% Date: 24/09/2025
%--------------------------------------------------------------------------

X = X - mean(X); %just to be sure, since it can disrupt results

% --- New input parsing ---
if nargin < 3
    settings = struct();
end
if ~isfield(settings, 'type')
    settings.type = 'rows&columns_imputation';
end
if ~isfield(settings, 'n_cross')
    settings.n_cross = 5;
end
if ~isfield(settings, 'tol')
    settings.tol = 1e-3;
end
if ~isfield(settings, 'max_iter')
    settings.max_iter = 100;
end
if ~isfield(settings, 'display')
    settings.display = 0;
end
% --------------------------

% Initialize cross-validation partition
switch settings.type
    case 'rows'
        c = cvpartition(size(X,1), 'KFold', settings.n_cross);
    case 'columns'
        c = cvpartition(size(X,2), 'KFold', settings.n_cross);
    case 'rows&columns_imputation'
        c = cvpartition(numel(X), 'KFold', settings.n_cross);
    case 'rows&columns_ALS'
        c = cvpartition(numel(X), 'KFold', settings.n_cross);
    case 'bi-cross'
        crows = cvpartition(size(X,1), 'KFold', floor(sqrt(settings.n_cross)));
        ccols = cvpartition(size(X,2), 'KFold', floor(sqrt(settings.n_cross)));
        cr = repmat([1:floor(sqrt(settings.n_cross))], 1, floor(sqrt(settings.n_cross)));
        cc = repelem([1:floor(sqrt(settings.n_cross))], floor(sqrt(settings.n_cross)));
end

R2 = [];  % Store R2 values for each fold and number of PCs
R2_null = [];  % Store R2 values for null model

% Loop over cross-validation folds
for cross = 1:settings.n_cross
    switch settings.type
        case 'bi-cross'
            if cross > numel(cr)
                break %all permutation of training sets have been tested yet
            else
                idx_train_rows = find(training(crows, cr(cross))); idx_test_rows = find(test(crows, cr(cross)));
                idx_train_cols = find(training(ccols, cc(cross))); idx_test_cols = find(test(ccols, cc(cross)));
                idx_test = sub2ind(size(X), repelem(idx_test_rows,length(idx_test_cols)), repmat(idx_test_cols,length(idx_test_rows),1) );
            end

        otherwise
            idx_train = training(c, cross);
            idx_test = test(c, cross);
    end

    % Loop over different numbers of PCs (k)
    for j = 1:n_dims_max
        switch settings.type
            case 'rows'
                Z = X(idx_train,:);
                [coeff, ~, ~] = pca(Z, "NumComponents", j);  % Compute k PCs
                real_values = X(idx_test,:);
                score =  real_values * coeff;
                predicted_values = score * coeff';

            case 'columns'
                Z = X(:,idx_train);
                [~, score] = pca(Z, "NumComponents", j);  % Compute k PCs
                real_values = X(:,idx_test);
                b = score\real_values;
                predicted_values = score * b;

            case 'rows&columns_imputation'
                Z = X;
                Z(idx_test) = 0;  % Set test set values to zero
                err = []; err(1) = NaN;
                
                for iter = 2:settings.max_iter+1
                    [coeff, score, ~] = pca(Z, "NumComponents", j);  % Compute k PCs
                    Z_hat = score * coeff';  % Reconstruct data with k PCs
                    err(iter) = sumsqr(Z_hat(idx_train) - X(idx_train));  % Compute error
                    if (err(iter-1) - err(iter)) / err(iter-1) < settings.tol
                        break
                    else
                        Z(idx_test) = Z_hat(idx_test);
                    end

                    if iter == settings.max_iter % If max iterations reached, display warning
                        disp('Stopping CV for reaching max iterations allowed');
                    end

                end

                real_values = X(idx_test);  % True values for test set
                predicted_values = Z_hat(idx_test);  % Predicted values for test set

            case 'rows&columns_ALS'
                Z = X;
                Z(idx_test) = NaN;  % Set test set values to NaN
                [coeff, score] = pca(Z, "NumComponents", j, "Algorithm","als");  % Compute k PCs with ALS for handling missing values
                tmp = score * coeff';
                real_values = X(idx_test);  % True values for test set
                predicted_values = tmp(idx_test);

            case 'bi-cross'
                X11 = X(idx_train_rows,idx_train_cols);
                X12 = X(idx_train_rows,idx_test_cols);
                X21 = X(idx_test_rows,idx_train_cols);
                real_values = X(idx_test_rows,idx_test_cols); %Completely HOLD-OUT data!

                [coeff1, score1] = pca(X11, "NumComponents", j);
                score2 = X21*coeff1;
                b = score1\X12;
                predicted_values = score2*b;

        end

        % Calculate R2 for PCA reconstruction error

        SST = sum((real_values - mean(real_values)).^2, 'all');  % Total sum of squares
        SSE = sum((real_values - predicted_values).^2, 'all');  % Sum of squares error


        R2(cross, j) = 1 - SSE / SST;  % R2 score for this fold and k PCs


        % Null model (randomly generated orthogonal components)
        d = size(X, 2);
        v = rand(d, 1);
        vbar = v.' / sumsqr(v);  % Normalize v
        Q = null(vbar).';  % Orthogonal complement of v
        orthoX = [vbar; Q];  % Create orthonormal basis

        % Null model reconstruction and R2 calculation
        score_null = X * orthoX(:, 1:j);
        X_rec = score_null * orthoX(:, 1:j)';  % Reconstruct with null model
        real_values = X(idx_test);
        predicted_values = X_rec(idx_test);
        SST = sum((real_values - mean(real_values)).^2,'all');
        SSE = sum((real_values - predicted_values).^2,'all');

        R2_null(cross, j) = 1 - SSE / SST;  % R2 score for null model

    end
end

% Compute average, standard deviation, and SEM for R2 scores
avg_R2 = mean(R2);  % Average R2 across folds
std_R2 = std(R2);  % Standard deviation of R2
SEM_R2 = std_R2 / sqrt(settings.n_cross);  % Standard error of the mean

% Find the best R2 score and its corresponding k (number of PCs)
[R2_best, idx_best] = max(avg_R2);

if idx_best == n_dims_max || idx_best == 1
    [R2_best, idx_best] = find_elbow(avg_R2);
end

% Find the index of the number of PCs where R2 is within one standard error from the best
idx_1SEfrombest = min(find(avg_R2 >= (R2_best - SEM_R2(idx_best))));

% R2 value corresponding to 1 SE from the best R2 score
R2_1SEfrombest = avg_R2(idx_1SEfrombest);

% Find the index of the number of PCs where R2 is within one standard dev from the best
idx_1STDfrombest = min(find(avg_R2 >= (R2_best - std_R2(idx_best))));

% R2 value corresponding to 1 SE from the best R2 score
R2_1STDfrombest = avg_R2(idx_1STDfrombest);

% Display results if requested
if settings.display == 1
    figure;
    errorbar(avg_R2, SEM_R2, 'k');  % Plot average R2 with error bars
    hold on;
    yline(R2_best, 'r');  % Line at best R2
    xline(idx_best, 'r');  % Line at best k
    yline(R2_1SEfrombest, 'g');  % Line at R2 corresponding to 1 SE from best
    xline(idx_1SEfrombest, 'g');  % Line at 1 SE from best k
    yline(R2_1STDfrombest, 'b');  % Line at R2 corresponding to 1 STD from best
    xline(idx_1STDfrombest, 'b');  % Line at 1 SE from best k
    legend({'avg cross-validated R2 ± SEM', 'BestR2','Best_id', '1 SE from best R2', '1 SE id', '1 STD from best', '1 STD id'});
    ylabel('crossvalidated R2')
    xlabel('Dimensions')
    title([ settings.type ' Cross-validation results'])
end

% Prepare output structure
crossval_PCA.avg_R2 = avg_R2;
crossval_PCA.std_R2 = std_R2;
crossval_PCA.SEM_R2 = SEM_R2;
crossval_PCA.R2_best = R2_best;
crossval_PCA.idx_best = idx_best;
crossval_PCA.R2_1SEfrombest = R2_1SEfrombest;
crossval_PCA.idx_1SEfrombest = idx_1SEfrombest;
crossval_PCA.R2_1STDfrombest = R2_1STDfrombest;
crossval_PCA.idx_1STDfrombest = idx_1STDfrombest;
crossval_PCA.R2_null = R2_null;


k = idx_best;

% Check the number of outputs requested and return accordingly
if nargout > 1
    varargout{1} = crossval_PCA;  % Return the crossval_PCA struct as the second output
end

    function [R2_best, idx_best] = find_elbow(V)
     
        if ~iscolumn(V)
            V = V';
        end

        %% Two lines method
        R2 = nan(size(V));
        for i = 2:length(V)-1
            x1 = [1:i]';
            mdl1 = fitlm(x1,V(x1),"linear");

            x2 = [i:length(V)]';
            mdl2 = fitlm(x2,V(x2),"linear");

            temp = mean([predict(mdl1, x1(end)); predict(mdl2, x2(1))]);
            Vhat = [predict(mdl1, x1(1:end-1)); temp; predict(mdl2, x2(2:end))];
            R2(i) = 1 - (sumsqr(V-Vhat)/sumsqr(V-mean(V)));
        end
        [~, idx_best] = max(R2);
        idx_best = idx_best;
        R2_best = V(idx_best);
    end

end

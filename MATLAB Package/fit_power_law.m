function [exponent, coefficients] = fit_power_law(A, display)
%--------------------------------------------------------------------------
% FIT_POWER_LAW Fits a power-law model to explained variance of PCs
%
% Computes PCA on input data and fits a power law model to the explained variance 
% of the first principal components (PCs). The model fitted is:
%    Y = a * X^(-b) + c
% where Y is explained variance and X is PC index.
%
% Syntax:
%   [exponent, coefficients] = fit_power_law(A)
%   [exponent, coefficients] = fit_power_law(A, display)
%
% Inputs:
%   A       - Data matrix (observations x variables)
%   display - (Optional) Logical flag to plot results (default: false)
%
% Outputs:
%   exponent     - Fitted exponent b of the power law
%   coefficients - Vector [a, b, c] containing all fitted parameters:
%                 a: scaling factor
%                 b: power law exponent
%                 c: offset term
%
% Example:
%   % Fit power law and display results
%   data = randn(1000, 50);
%   [b, coeff] = fit_power_law(data, true);
%
% Author: Francesco E. Vaccari, PhD
% Date: September 24, 2025
%--------------------------------------------------------------------------

  % 1) compute explained variance of first PCs
  num_PCs = min([size(A,2), 25]);
  [~,~,~,~,Y,~] = pca(A, "NumComponents", num_PCs);
  Y = Y(1:num_PCs);
  if size(Y,2) > size(Y,1)
      Y = Y';
  end

  % 2) x‐axis = PC index
  X = (1:numel(Y))';

  % 3) pack into table for fitnlm
  tbl = table(X, Y);

  % 4) define power‐law model: Y = a * X^(-b) + c
  modelfun = @(b, x) b(1) .* x(:,1).^-b(2) + b(3);
  

  % 5) initial guesses for [a, b, c]
  beta0 = [Y(1), 1, min(Y)];

  % 6) fit
  mdl = fitnlm(tbl, modelfun, beta0);

  % 7) extract
  coefficients = mdl.Coefficients.Estimate;   % [a; b; c]
  exponent     = coefficients(2);

  % 8) optional plot
  if display
      fittedY = coefficients(1) .* X.^(-coefficients(2)) + coefficients(3);
      figure;
      plot(X, Y, 'bo', 'MarkerSize', 8, 'LineWidth', 1.5);
      hold on;
      plot(X, fittedY, 'r-', 'LineWidth', 2);
      grid on;
      xlabel('PC index');
      ylabel('Explained variance (%)');
      title(sprintf('Power Law Fit: Y = %.2f·X^{-%0.2f} + %.2f', ...
                    coefficients(1),coefficients(2),coefficients(3)));
      legend('data','fit','Location','northeast');
  end

end

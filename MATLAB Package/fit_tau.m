function [tau, coefficients] = fit_tau(A, display)
% fit_lambda - Fits an exponential model to explained variance of the first PCs 
% of the input data A and returns the fitted time constant (tau).
%
% This function fits the model Y = a * exp(-b * X) + c to the explained variance Y, where X is the 
% sequence of PCs (1:length(Y)), and a, b, and c are the parameters of the model.
% The fitting is performed using the `fitnlm` function. Optionally, the function can 
% display a plot of the noisy data and the fitted curve.
%
% Inputs:
%   A        - Data matrix (rows are observations, columns are variables)
%   display  - A logical flag (true/false) to control whether to display the plot.
%
% Outputs:
%   tau             - The fitted parameter b from the exponential model (time constant).
%   coefficients    - A 3-element vector containing the fitted parameters [a, b, c] of the model.
%
% Example:
%   [tau, coeff] = fit_lambda(Y, true); 
%   This will return tau (the time constant) and the full set of fitted coefficients.
%
% Author: [Francesco E. Vaccari, PhD]
% Date: [12/11/2024]

[~,~,~,~,explained,~] = pca(A);
Y = explained(1:min([size(A,2) 25]));

% Ensure Y is a column vector.
if size(Y,2) > size(Y,1)
    Y = Y';
end

% Create the X data points.
X = (1:length(Y))';

% Create the table for nonlinear regression.
tbl = table(X, Y);

% Define the exponential model: Y = a * exp(-b * X) + c
modelfun = @(b,x) b(1) * exp(-b(2)*x(:, 1)) + b(3);  

% Initial guess for the parameters: [a, b, c]
beta0 = [0, 0.5, 0]; 

% Fit the nonlinear model using fitnlm.
mdl = fitnlm(tbl, modelfun, beta0);

% Extract the fitted coefficients.
coefficients = mdl.Coefficients{:, 'Estimate'};

% Extract the second coefficient (b), which is the time constant tau.
tau = coefficients(2);

% Optionally display the results.
if display
    figure;
    plot(X, Y, 'b*', 'LineWidth', 2, 'MarkerSize', 15); % Plot the noisy data.
    hold on;
    plot(X, coefficients(1) * exp(-coefficients(2) * X) + coefficients(3), 'r-', 'LineWidth', 2); % Plot the fitted curve.
    grid on;
    title(['Exponential Regression / tau = ' num2str(tau)]);
    xlabel('First PCs');
    ylabel('Explained Variance');
    legend('Real Y', 'Fitted Y', 'Location', 'north');
    legend('FontSize', 12);
    formulaString = sprintf('Y = %.2f * exp(-%.2f * X) + %.2f', coefficients(1), coefficients(2), coefficients(3));
    text(7, 11, formulaString, 'FontSize', 12, 'FontWeight', 'bold');
end

end

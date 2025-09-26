function real_latents = generate_random_dynamical_series(num_latent, time_series_length)
% GENERATE_RANDOM_DYNAMICAL_SERIES Generates a random time series 
% from a dynamical system defined by either a noisy 3-variable Lorenz system or 
% a noisy Lorenz96-like system for higher dimensions.
%
%   real_latents = generate_random_dynamical_series(num_latent, time_series_length)
%
% Inputs:
%   num_latent         - Number of state variables for the dynamical system.
%                        If num_latent == 3, a modified classic Lorenz system is used.
%                        Otherwise, a Lorenz96 system formulation is used.
%                        NOTE: num_latent must be >= 3
%   time_series_length - Number of time points after sampling with a fixed dt.
%
% Outputs:
%   real_latents       - trajectory data produced by evolving the chosen dynamical system.
%
% The function uses MATLAB's ode45 to solve the ODEs. A random forcing parameter 
% F is generated, and the initial condition u0 is set as a constant vector, with a 
% perturbation at the first coordinate. In the case of a 3-variables Lorenz
% system, parameters are fixed.
%
% Example:
%   real_latents = generate_random_dynamical_series(5, 3000);

% Author: [Francesco E. Vaccari, PhD]
% Date: [24/09/2025]

F = randi(10);
u0 = F * ones(num_latent,1);
u0(1) = u0(1) + randi(10);  

% Time settings
dt = 0.01;          % sampling time
Tf = dt*time_series_length;          % final time
tspan = 0:dt:(Tf-dt);    % time vector

% Solve the ODE using ode45. Note: by providing tspan we force output at dt intervals.
[T, Y] = ode45(@(t,x) dynamical_sys(t,x,F), tspan, u0);

real_latents = Y;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Dynamical system function definition
    function dx = dynamical_sys(~, x, F)
        % Determine the system size
        num_latent = length(x);
        dx = zeros(num_latent,1);

        if num_latent == 3 %using normal Lorenz system with 3 variables

            s = 10; %default parameters
            p = 28;
            b = 8/3;
            dx(1) = s*(x(2)-x(1)) +normrnd(0,2*s);
            dx(2) = x(1)*(p-x(3)) - x(2) +normrnd(0,2*s);
            dx(3) = x(1)*x(2) - b*x(3) +normrnd(0,2*s);

        else %using lorenz96 system for more variables

            % Edge cases handled explicitly for performance
            dx(1)   = (x(2) - x(num_latent-1)) * x(num_latent)   - x(1)   + F +normrnd(0,2*F);
            dx(2)   = (x(3) - x(num_latent))   * x(1)   - x(2)   + F +normrnd(0,2*F);
            dx(num_latent)   = (x(1) - x(num_latent-2)) * x(num_latent-1) - x(num_latent)   + F +normrnd(0,2*F);

            % The general case for indices 3 to N-1
            for n = 3:(num_latent-1)
                dx(n) = (x(n+1) - x(n-2)) * x(n-1) - x(n) + F +normrnd(0,2*F);
            end
        end
    end
end
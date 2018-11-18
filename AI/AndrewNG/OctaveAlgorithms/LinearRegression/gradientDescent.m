function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
% Perform gradient descent for multiple variables to find theta, that 
%   minimizes cost function J perfectly.
  for iter = 1:num_iters
    theta = theta - alpha * (1 / m) * (((theta' * X')' - y )' * X )';
  end
end

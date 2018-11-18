function [theta] = normalEquation(X, y)
% Use normal equation to solve linear regression problem in one step.
theta = inv(X' * X) * X' * y;
end

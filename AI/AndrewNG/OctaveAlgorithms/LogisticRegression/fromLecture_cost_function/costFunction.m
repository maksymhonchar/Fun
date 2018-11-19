function [jVal, gradient] = costFunction(theta)
  % Example: apply CF to function J(theta)=(theta1-5)^2+(theta2-5)^2
  jVal = (theta(1) - 5)^2 + (theta(2) - 5)^2;
  gradient = zeros(2, 1);
  gradient(1) = 2 * (theta(1) - 5);  % Partial derivative of Theta1.
  gradient(2) = 2 * (theta(2) - 5);  % Partial derivative of Theta2.
endfunction

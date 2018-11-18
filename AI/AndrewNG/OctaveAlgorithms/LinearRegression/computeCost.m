function J = computeCost(X, y, theta)
  % Compute cost function for linear regression with multiple variables.
  m = length(y);  % number of training examples
  J = 0;
  
  % Approach using sum function.
  % J = 1 / (2 * m) * sum( ((theta' * X')' - y) .^ 2 );

  % Vectorized approach.
  J = 1 / (2 * m) * (X * theta - y)' * (X * theta - y);

end

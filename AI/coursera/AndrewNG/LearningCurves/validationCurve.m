function [lambda_vec, error_train, error_val] = ...
    validationCurve(X, y, Xval, yval)

lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';

error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);

for i = 1:length(lambda_vec),
  lambda = lambda_vec(i);
  
  [theta] = trainLinearReg(X, y, lambda);
	
  [error_train(i), grad] = linearRegCostFunction(X, y, theta, 0);  % note: lambda=0 -- no regularization
	
  [error_val(i), grad] = linearRegCostFunction(Xval , yval, theta, 0);	% note: lambda=0 -- no regularization
end

end

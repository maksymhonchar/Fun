function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)

m = size(X, 1);

error_train = zeros(m, 1);
error_val   = zeros(m, 1);

% ----- %

val_size = size(Xval,1);

for i = 1:m,
  [theta] = trainLinearReg([ones(i , 1) X(1:i , :)], y(1:i), lambda);
	
  [error_train(i), grad] = linearRegCostFunction(
    [ones(i , 1) X(1:i, :)],
    y(1:i), theta, 0  % note: lambda=0 -- no regularization
  );
	
  [error_val(i), grad] = linearRegCostFunction(
    [ones(val_size , 1) Xval],
    yval, theta, 0  % note: lambda=0 -- no regularization
  );	
end

% ----- %

end

function [J, grad] = linearRegCostFunction(X, y, theta, lambda)

m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));

% ---- %

H = X*theta;
J = sum((H - y).^2) / (2 * m) + lambda*sum(theta(2:end).^2) / (2 * m);  % cost

grad = X'*(H - y) / m + lambda*[0;theta(2:end)] / m;  % gradient

% ---- %

grad = grad(:);

end

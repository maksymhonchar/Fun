% Not sure if this implementation is even correct tbh
function theta = grad_descent(X, y)
  iterations = 2000;
  alpha = 0.001;
  m = size(X, 1);
  n = size(X, 2);
  theta = zeros(n, 1);
  X = [ ones(m, 1), X ];
  
  theta = theta - alpha * ( (1/m) * sum( (theta' * X - y)' * X ) );
  
  theta = [ ones(n, 1), theta ];
endfunction

function Z = projectData(X, U, K)

Z = zeros(size(X, 1), K);

% ---- %

% slow:
% Ureduce = U(:,1:K);
% Z = Ureduce'*X';

for i = 1:size(X,1),

  x = X(i, :)';

  for j = 1:K,
    projection_k = x' * U(:, j);
    Z(i,j) = projection_k;
  end

end    

% ---- %

end

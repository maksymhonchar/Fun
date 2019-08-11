function X_rec = recoverData(Z, U, K)

X_rec = zeros(size(Z, 1), size(U, 1));

% ---- %

for i = 1:size(Z,1),

  v = Z(i,:)';

  for j = 1:size(U,1),
    recovered_j = v' * U(j, 1:K)';  % src: from the course
    X_rec(i,j) = recovered_j;
  end
  
end  

% ---- %

end

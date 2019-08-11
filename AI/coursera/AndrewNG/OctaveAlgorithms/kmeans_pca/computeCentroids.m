function centroids = computeCentroids(X, idx, K)

[m n] = size(X);

centroids = zeros(K, n);

% ----- %

counts = zeros(K,1);

for i = 1:m,
  for j = 1:K,
    if idx(i) == j,
      centroids(j,:) = (centroids(j,:) + X(i,:));
      counts(j) = counts(j) + 1;
    end
  end
end  

centroids = centroids./counts;

% ----- %

end


function centroids = kMeansInitCentroids(X, K)

centroids = zeros(K, size(X, 2));

% ---- %

% Initialization of centroids - randomly from the existing training set
randidx = randperm(size(X, 1));

% First k examples are our centroids
centroids = X(randidx(1:K), :);

% ---- %

end


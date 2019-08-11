function idx = findClosestCentroids(X, centroids)

K = size(centroids, 1);

idx = zeros(size(X,1), 1);

% ----- %

m = size(X,1);

for i = 1:m,
  minDistance = 10^6; % randomly chosen large enough value 

  for j = 1:K,
    distance = sum((X(i,:) - centroids(j,:)).^2);
    
    if distance < minDistance,
      minDistance = distance;
      idx(i) = j;
    end
    
  end

end  

% ----- %

end


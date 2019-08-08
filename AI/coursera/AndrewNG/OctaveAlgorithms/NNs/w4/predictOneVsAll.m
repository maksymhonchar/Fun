function p = predictOneVsAll(all_theta, X)

m = size(X, 1);
num_labels = size(all_theta, 1);

p = zeros(size(X, 1), 1);

X = [ones(m, 1) X];

% ------- %

prob_mat = X * all_theta';
[prob, p] = max(prob_mat,[],2);

end

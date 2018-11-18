function [X_norm, mu, sigma] = featureNormalize(X)
% Normalize features for X design matrix.
% Normalization: mean normalization + standard deviation normalization.
% NOTE, that X is a matrix with separate features in each column -> apply
%   normalization for each feature == for each column in design matrix X.

% Initial values.
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% Apply mean normalization.
mu = mean(X);
X_norm = X_norm - mu;

% Apply standard deviation normalization.
sigma = std(X);
X_norm = X_norm ./ sigma;

end

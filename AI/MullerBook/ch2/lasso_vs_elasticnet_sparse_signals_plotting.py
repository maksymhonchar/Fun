import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso, ElasticNet
from sklearn.metrics import r2_score

np.random.seed(99)

# Generate {X,y}
n_samples, n_features = 50, 200
X = np.random.randn(n_samples, n_features)
coef = 3 * np.random.randn(n_features)
inds = np.arange(n_features)
np.random.shuffle(inds)
coef[inds[10:]] = 0  # sparsify coef
y = np.dot(X, coef)
y += 0.01 * np.random.normal(size=n_samples) # noise

# Create train/test sets
n_samples = X.shape[0]
X_train, y_train = X[:n_samples // 2], y[:n_samples // 2]
X_test, y_test = X[n_samples // 2:], y[n_samples // 2:]

# Lasso method
lasso = Lasso(alpha=0.1)
# Make a prediction
y_pred_lasso = lasso.fit(X_train, y_train).predict(X_test)
r2_score_lasso = r2_score(y_test, y_pred_lasso)
lasso_train_score_scoremethod = lasso.score(X_train, y_train)
# print(lasso)
print('r2 on test data: {0}'.format(r2_score_lasso))
print(lasso_train_score_scoremethod)

print("")

# ElasticNet method
enet = ElasticNet(alpha=0.1, l1_ratio=0.7)
# Make a prediction
y_pred_enet = enet.fit(X_train, y_train).predict(X_test)
r2_score_enet = r2_score(y_test, y_pred_enet)
enet_train_score_scoremethod = enet.score(X_train, y_train)
# print(enet)
print('r2 on test data: {0}'.format(r2_score_enet))
print(enet_train_score_scoremethod)

# Plot coef_-s
plt.plot(enet.coef_, color='lightgreen', linewidth=2, label='Elastic net coefficients')
plt.plot(lasso.coef_, color='gold', linewidth=2, label='Lasso coefficients')
plt.plot(coef, '--', color='navy', label='original coefficients')
plt.legend(loc='best')
plt.title("Lasso R^2: %f, Elastic Net R^2: %f" % (r2_score_lasso, r2_score_enet))
plt.show()

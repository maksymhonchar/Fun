import mglearn

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

# OLS method - ordinary least squares - the simplest and most classic linear method for regression.
X, y = mglearn.datasets.make_wave(n_samples=100)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)

# The 'slope' parameters (w) - weights or coefficients:
print("lr.coef_: {0}".format(lr.coef_))
# The 'offset' or intercept (b):
print("lr.intercept_: {0}".format(lr.intercept_))

# import matplotlib.pyplot as plt
# mglearn.plots.plot_linear_regression_wave()
# plt.show()

# Get test set performance:
print('Training set score: {:.2f}'.format(lr.score(X_train, y_train)))
print('Test set score: {:.2f}'.format(lr.score(X_test, y_test)))

# Try boston dataset: 506 samples & 105 derived features
X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
lr = LinearRegression()
lr.fit(X, y)
print('Training set score: {:.2f}'.format(lr.score(X_train, y_train)))
print('Testing set score: {:.2f}'.format(lr.score(X_test, y_test)))

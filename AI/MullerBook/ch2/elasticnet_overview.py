from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.datasets import make_regression


X, y = make_regression(n_features=75, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

en_regr = ElasticNet(random_state=0)
en_regr.fit(X_train, y_train)

ridge_regr = Ridge()
ridge_regr.fit(X_train, y_train)

lasso_regr = Lasso()
lasso_regr.fit(X_train, y_train)

print("ELASTICNET")
print("training set score: {:.2f}".format(en_regr.score(X_train, y_train)))
print("test set score: {:.2f}".format(en_regr.score(X_test, y_test)))
# print("coef_: {0}".format(en_regr.coef_))

print("RIDGE")
print("training set score: {:.2f}".format(ridge_regr.score(X_train, y_train)))
print("test set score: {:.2f}".format(ridge_regr.score(X_test, y_test)))
# print("coef_: {0}".format(ridge_regr.coef_))

print("LASSO")
print("training set score: {:.2f}".format(lasso_regr.score(X_train, y_train)))
print("test set score: {:.2f}".format(lasso_regr.score(X_test, y_test)))
# print("coef_: {0}".format(lasso_regr.coef_))

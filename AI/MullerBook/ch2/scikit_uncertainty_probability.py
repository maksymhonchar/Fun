import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_circles, load_iris
from sklearn.model_selection import train_test_split


X, y = make_circles(noise=.25, factor=.5, random_state=1)

# Rename the classes 'blue' and 'red' for illustration purposes.
y_named = np.array( ['blue', 'red'] )[y]

X_train, X_test, y_train_named, y_test_named, y_train, y_test = train_test_split(
    X, y_named, y,
    random_state=42
)

gbrt = GradientBoostingClassifier(random_state=42)
gbrt.fit(X_train, y_train_named)


def decision_fun():
    # Decision value encodes how strongly the model believes a data point to belong to the 'positive' class, in this case class 1. 
    # Positive values indicate a preference for the positive class, vice versa for the negative values ("other" class).
    print('X_test.shape():{0}'.format(X_test.shape))  # (25, 2)
    print('decision function shape: {0}'.format(gbrt.decision_function(X_test).shape))  # (25,)
    print(gbrt.decision_function(X_test)[:6])
    print("Thresholded decision function:\n{}".format(gbrt.decision_function(X_test) > 0))
    print("Predictions:\n{}".format(gbrt.predict(X_test)))
    
    # make the boolean True/False into 0 and 1
    greater_zero = (gbrt.decision_function(X_test) > 0).astype(int)
    # use 0 and 1 as indices into classes_
    pred = gbrt.classes_[greater_zero]
    # pred is the same as the output of gbrt.predict
    print("pred is equal to predictions: {0}".format(np.all(pred == gbrt.predict(X_test))))
    
    decision_function = gbrt.decision_function(X_test)
    print("Decision function minimum: {:.2f} maximum: {:.2f}".format(
        np.min(decision_function), np.max(decision_function))
    )  # [-7.75; 4.11]

# decision_fun()


def probability_fun():
    # is often mroe easily understood than the output of decision_function
    # first entry - estimated probability of the first class
    # second entry - estimated probability of the second class
    print("Shape of probabilities:\n{0}".format(gbrt.predict_proba(X_test).shape))  # (25, 2)
    print("Predicted probabilities:\n{0}".format(gbrt.predict_proba(X_test[:6])))
    # A model is called calibrated if the reported uncertainty actually matches how correct 
    # it is—in a calibrated model, a pre diction made with 70% certainty would be correct 70% 
    # of the time.

# probability_fun()


def uncertainty_multiclass_clf():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=42)
    gbrt = GradientBoostingClassifier(learning_rate=0.01, random_state=0)
    gbrt.fit(X_train, y_train)
    print("Decision function shape: {}".format(gbrt.decision_function(X_test).shape))  # (38, 3)
    print("Decision function:\n{}".format(gbrt.decision_function(X_test)[:6, :]))

    # recover predictions from these scores by finding max entry for each data point:
    print('Argmax of decision function:\n{0}'.format(np.argmax(gbrt.decision_function(X_test), axis=1)))
    print('Predictions:\n{0}'.format(gbrt.predict(X_test)))

    # Probabilities:
    print("Predicted probabilities:\n{}".format(gbrt.predict_proba(X_test)[:6]))
    print("Sums: {}".format(gbrt.predict_proba(X_test)[:6].sum(axis=1)))  # Sums: [ 1. 1. 1. ... 1.]

    print("Argmax of predicted probabilities:\n{}".format(np.argmax(gbrt.predict_proba(X_test), axis=1)))
    print("Predictions:\n{}".format(gbrt.predict(X_test)))


uncertainty_multiclass_clf()

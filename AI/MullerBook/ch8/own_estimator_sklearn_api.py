# code src: copy-paste from Muller's book.

# usage: custom data preprocessing (if preprocessing is data dependent)

# simplest transformer: 1) inherit from BaseEstimator & TransformerMixin @ 2)__init__(), fit(), predict()

from sklearn.base import BaseEstimator, TransformerMixin


class MyTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, param_a, param_b):
        # All parameters must be specified in the __init__ function

        # todo: throw exception if param_a=None OR param_b=None

        self.param_a = -1
        self.param_b = -1

    def fit(self, X, y=None):
        # .fit() should only take X and y as parameters
        # NOTE: Even if your model is unsupervised, you need to accept a "y" argument

        # Fit the model
        # .fit()
        print('fitting the model ... ... ...')

        # .fit() returns self
        return self

    def transform(self, X):
        # .transform() takes X as the only parameter

        # Apply transformation to X
        X_transformed = X + 1
        print('transforming X... ... ...')

        # .predict() returns trasnformed data
        return X_transformed

# Implementing a classifier or regressor works similarly, only instead of Transformer Mixin you need to inherit from ClassifierMixin or RegressorMixin
# Also, instead of implementing transform , you would implement predict .

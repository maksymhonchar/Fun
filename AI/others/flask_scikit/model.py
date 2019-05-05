# develop & train model

import json
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import request


dataset = pd.read_csv(
    '/home/max/Documents/learn/learnai/flask_scikit/salary_dataset.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.33,
    random_state=0
)

regressor = LinearRegression(n_jobs=-1)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

with open('regr_model.pkl', 'wb') as f:
    pickle.dump(regressor, f)

#!/usr/bin/env python
# coding: utf-8

# In[1]:


# src: https://towardsdatascience.com/hands-on-predict-customer-churn-5c2a42806266


# In[2]:


# Churn quantifies the number of customers who have
# unsubscribed or canceled their service contract.

# Steps

# 1. Use Case / Business Case
# Only by understanding the final objective we can build
# a model that is actually of use.

# 2. Data collection & cleaning
# identify the right data sources, cleansing the data sets and
# preparing for feature selection or engineering.

# 3. Feature selection & engineering
# decide which features we want to include in our model and
# prepare the cleansed data to be used for the machine learning
# algorithm to predict customer churn.

#  4. Modelling
# Find the right model (selection) and evaluate that the
# algorithm actually works.

# 5. Insights and Actions
# Evaluate and interpret the outcomes
# In our case we actually want to make them stop leaving.


# In[3]:


# Load libraries

import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.figsize'] = [5.0, 5.0]

import numpy as np

import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

from sklearn.model_selection import train_test_split, validation_curve, learning_curve
from sklearn import metrics

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


# In[4]:


# Load dataset

# WA_Fn-UseC_-Telco-Customer-Churn.csv

data = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

data = data.drop(['customerID'], axis=1)


# In[5]:


display(data.shape)
display(data.head(5))

display(data.describe(include='all'))
display(data.dtypes)

display(data.isnull().sum())
display(data.isnull().sum().sum())


# In[6]:


data['Churn'].hist()
plt.show()

data['Churn'].value_counts(sort=True).plot(kind='pie', autopct='%1.1f%%')
plt.title("% of churn in dataset")
plt.show()


# In[7]:


# Remove those with tenure=0: 11 rows
# Reason: these rows don't have TotalCharges value (instead - " " value)
display(data[data['tenure'] == 0])
totalcharges_empty_indices = data[data['tenure'] == 0]
data = data.drop(totalcharges_empty_indices.index)


# In[8]:


# Fix 'TotalCharges' column dtype

display(data['TotalCharges'].dtype)  # object
data['TotalCharges'] = data['TotalCharges'].astype(float)
display(data['TotalCharges'].dtype)  # float64


# In[9]:


# Check for invalid 'TotalCharges' or invalid 'MonthlyCharges' values

display(data[data['MonthlyCharges'] > data['TotalCharges']])  # should be empty


# In[10]:


# Convert categorical columns into numerical columns
# Use get_dummies for that

data_dummies = data.copy()

data_churn = data_dummies['Churn']
data_dummies = data_dummies.drop('Churn', axis=1)

data_dummies = pd.get_dummies(data_dummies)  # todo: try with drop_first=True

display(data.shape, data_dummies.shape)


# In[11]:


# Split the dataset

y = data_churn.values
X = data_dummies

TEST_SIZE = 0.2

X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)


# In[12]:


# Build a model

model = LogisticRegression(solver='newton-cg')
result = model.fit(X_tr, y_tr)


# In[13]:


# Evaluate model

y_pred = model.predict(X_val)

display(metrics.accuracy_score(y_val, y_pred))

# accuracy:
# 0.7882018479033405 without drop_first
# 0.7874911158493249 with drop_first


# In[14]:


# Look at weights of our model

logreg_weights = pd.Series(model.coef_[0], index=X.columns.values)
display(logreg_weights.sort_values(ascending=False))


# In[33]:


# Plot validation curves

params_C_svc = np.array([0.001, 0.01, 0.1, 0.5, 1, 10, 50, 100, 1000])

train_scores, valid_scores = validation_curve(
    SVC(kernel='rbf', gamma='scale'),
    X, y,
    'C', params_C_svc,
    cv=5,
    verbose=5, n_jobs=-1
)


# In[34]:


val_scores = []

plt.plot(params_C_svc, [cv_acc.mean() for cv_acc in train_scores])
plt.plot(params_C_svc, [cv_acc.mean() for cv_acc in valid_scores])
plt.show()

display([cv_acc.mean() for cv_acc in train_scores])
display([cv_acc.mean() for cv_acc in valid_scores])


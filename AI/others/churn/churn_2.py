#!/usr/bin/env python
# coding: utf-8

# In[1]:


# src: http://datareview.info/article/prognozirovanie-ottoka-klientov-so-scikit-learn/


# In[ ]:


# Показатель оттока клиентов – бизнес-термин, описывающий
# насколько интенсивно клиенты покидают компанию или
# прекращают оплачивать товары или услуги.

# Это ключевой показатель для многих компаний, потому что
# зачастую приобретение новых клиентов обходится намного дороже,
# чем удержание старых (в некоторых случаях от 5 до 20 раз дороже).

# Примеры использования:
# 1. мобильные операторы, операторы кабельного телевидения и
# компании, обслуживающие прием платежей с помощью кредитных карт
# 2. казино используют прогнозные модели, чтобы предсказать
# идеальные условия в зале, позволяющие удержать игроков
# в Блэкджек за столом.
# 3. Aвиакомпании могут предложить клиентам, у которых есть
# жалобы, заменить их билет на билет первого класса.

# Эффективное удержание клиентов сводится к задаче, в рамках
# которой, используя имеющиеся данные, необходимо отличить
# клиентов, собирающихся уйти, от тех, кто этого делать
# не собирается.


# In[ ]:


# datset src: https://raw.githubusercontent.com/michaelulin/churn/master/work/churn_model/data/churn.csv


# In[88]:


# Load libraries

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import KFold, train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


# In[3]:


# Load dataset

raw_churn_df = pd.read_csv('churn.csv')


# In[17]:


display(raw_churn_df.shape)

display(raw_churn_df.head(), raw_churn_df.tail())

display(raw_churn_df.columns.values)

display(raw_churn_df.dtypes)

display(raw_churn_df.isnull().sum())


# In[78]:


# Isolate target data

y = raw_churn_df['Churn?']
X = raw_churn_df.drop('Churn?', axis=1)


# In[79]:


# Drop irrelevant features

features_to_drop = ['State', 'Area Code', 'Phone']
X = X.drop(features_to_drop, axis=1)


# In[80]:


# Encode yes/no with 1/0 values

X["Int'l Plan"] = X["Int'l Plan"].map({'no': 0, 'yes': 1})
X["VMail Plan"] = X["VMail Plan"].map({'no': 0, 'yes': 1})


# In[81]:


# Scale everything

std_scaler = StandardScaler(with_mean=True)
X = std_scaler.fit_transform(X)

display(X.shape)


# In[90]:


# Perform CV for SVM, random forest and kNN

def try_clf(X, y, clf_nofit):
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, random_state=42)
    
    clf = clf_nofit.fit(X_tr, y_tr)
    
    y_pred = clf.predict(X_val)
    
    display(clf_nofit.__class__.__name__)
    
    display(accuracy_score(y_val, y_pred))
    
    display(confusion_matrix(y_val, y_pred))
    
    display("prec, rec, f1, support", precision_recall_fscore_support(y_val, y_pred))
    
try_clf(X, y, SVC(gamma='scale'))
try_clf(X, y, RandomForestClassifier(n_estimators=100, n_jobs=-1))
try_clf(X, y, KNeighborsClassifier())

# std scaler with_mean=False accuracies:
# 0.9256594724220624
# 0.9484412470023981
# 0.8896882494004796

# std scaler with_mean=True accuracies:
# 0.9256594724220624
# 0.9496402877697842
# 0.8896882494004796


# In[86]:


# Recall
# Каково отношение количества правильно спрогнозированных уходов
# к общему количеству фактических уходов?

# Precision
# Каково отношение количества правильно спрогнозированных уходов
# к общему количеству спрогнозированных уходов?


# In[101]:


# # Predict probabilities

# def try_probab(X, y, clf_nofit):
#     X_tr, X_val, y_tr, y_val = train_test_split(X, y, random_state=42)
    
#     clf = clf_nofit.fit(X_tr, y_tr)
    
#     y_prob = clf.predict_proba(X_val)
    
# #     for i in range(len(X)):
# #         display("y_true={0}, Predicted={1}".format(y[i], y_prob[i]))

#     display(pd.value_counts(y_prob[:, 1]))
 
# try_probab(X, y, SVC(gamma='scale', probability=True))
# # try_probab(X, y, RandomForestClassifier(n_estimators=100, n_jobs=-1))
# # try_probab(X, y, KNeighborsClassifier())


# # for i in range(len(Xnew)):
# # 	print("X=%s, Predicted=%s" % (Xnew[i], ynew[i]))


# In[ ]:


# todo: calibration and discrimination
    
# https://github.com/ghuiber/churn/blob/master/churn_measurements.py

# from churn_measurements import calibration, discrimination


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gc
gc.collect()


# In[2]:


# Load libraries

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, multilabel_confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb
import lightgbm as lgb


# In[3]:


# Load datasets

TRAIN_FILEPATH = 'data/train.csv'
train_df = pd.read_csv( TRAIN_FILEPATH, header=0 )
display(train_df.shape)


# In[4]:


# Split training set on train/validation sets (60-20-20, 70-30)

# save target value
train_label = train_df['Cover_Type']  # {1;2;...;6;7}, 2160 entires each
train_df = train_df.drop( ['Id', 'Cover_Type'], axis=1 )

# train_test_split

VALIDATION_SIZE = 0.3

X_tr, X_val, y_tr, y_val = train_test_split(
    train_df, train_label,
    test_size=VALIDATION_SIZE,
    shuffle=True
)


# In[5]:


# 1. sklearn.LogisticRegression, l2 regularization

logreg_l2_model = LogisticRegression(
    C=1.0,
    solver='lbfgs',  # multinomial loss
    penalty='l2',
    max_iter=1000,
    multi_class='multinomial',
    n_jobs=-1
)

logreg_l2_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg_model', logreg_l2_model)
])

get_ipython().run_line_magic('time', 'logreg_l2_pipeline.fit( X_tr, y_tr )')


# In[6]:


get_ipython().run_line_magic('time', 'logreg_y_val_pred = logreg_l2_pipeline.predict( X_val )')


# In[7]:


display( accuracy_score(y_val, logreg_y_val_pred) )

print( classification_report(y_val, logreg_y_val_pred) )

display( multilabel_confusion_matrix(y_val, logreg_y_val_pred) )


# In[8]:


# 2. sklearn.LogisticRegression, l1 regularization

logreg_l1_model = LogisticRegression(
    C=1.0,
    solver='liblinear',  # one-vs-rest scheme
    penalty='l1',
    max_iter=1000,
    multi_class='ovr'  # can't use 'multinomial'
)

logreg_l1_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg_model', logreg_l1_model)
])

get_ipython().run_line_magic('time', 'logreg_l1_pipeline.fit( X_tr, y_tr )')


# In[9]:


get_ipython().run_line_magic('time', 'logreg_y_val_pred = logreg_l1_pipeline.predict( X_val )')


# In[10]:


display( accuracy_score(y_val, logreg_y_val_pred) )

print( classification_report(y_val, logreg_y_val_pred) )

display( multilabel_confusion_matrix(y_val, logreg_y_val_pred) )


# In[27]:


# 3. sklearn.RandomForestClassifier

rfc_model = RandomForestClassifier(
    n_estimators=1000,
    n_jobs=-1
)

get_ipython().run_line_magic('time', 'rfc_model.fit( X_tr, y_tr )')


# In[28]:


get_ipython().run_line_magic('time', 'rfc_y_val_pred = rfc_model.predict( X_val )')


# In[29]:


display( accuracy_score(y_val, rfc_y_val_pred) )

print( classification_report(y_val, rfc_y_val_pred) )

display( multilabel_confusion_matrix(y_val, rfc_y_val_pred) )


# In[14]:


display(
    pd.DataFrame({
        'feature_name': X_tr.columns,
        'feature_imp': rfc_model.feature_importances_
    }).sort_values( by='feature_imp', ascending=False ).head()
)


# In[15]:


# 4. xgb.XGBClassifier

eval_set = [ (X_val, y_val) ]

xgb_model = xgb.XGBClassifier(
    gamma=0.025,
    learning_rate=0.35,
    max_depth=5,
    n_estimators=1000,
    objective='multi:softmax',
    n_jobs=4
)

get_ipython().run_line_magic('time', "xgb_model.fit( X_tr, y_tr, eval_set=eval_set, eval_metric='merror', verbose=False )")


# In[16]:


get_ipython().run_line_magic('time', 'xgb_y_val_pred = xgb_model.predict( X_val )')


# In[17]:


display( accuracy_score(y_val, xgb_y_val_pred) )

print( classification_report(y_val, xgb_y_val_pred) )

display( multilabel_confusion_matrix(y_val, xgb_y_val_pred) )


# In[18]:


display(
    pd.DataFrame({
        'feature_name': X_tr.columns,
        'feature_imp': xgb_model.feature_importances_
    }).sort_values( by='feature_imp', ascending=False ).head()
)


# In[19]:


# 5. lgb.LGBMClassifier

lgb_model = lgb.LGBMClassifier(
    learning_rate=0.2,
    max_depth=-1,
    n_estimators=1000,
    objective='multiclass',
    n_jobs=8,
    verbose=0
)

get_ipython().run_line_magic('time', 'lgb_model.fit( X_tr, y_tr )')


# In[20]:


get_ipython().run_line_magic('time', 'lgb_y_val_pred = lgb_model.predict( X_val )')


# In[21]:


display( accuracy_score(y_val, lgb_y_val_pred) )

print( classification_report(y_val, lgb_y_val_pred) )

display( multilabel_confusion_matrix(y_val, lgb_y_val_pred) )


# In[22]:


display(
    pd.DataFrame({
        'feature_name': X_tr.columns,
        'feature_imp': lgb_model.feature_importances_
    }).sort_values( by='feature_imp', ascending=False ).head()
)


# In[23]:


##################################################################################


# In[33]:


# Blend baseline LGBclf, XGBclf and sklearn RandomForestClassifier models into submission

# Load test set
TEST_FILEPATH = 'data/test.csv'
test_df = pd.read_csv( TEST_FILEPATH, header=0 )
display(test_df.shape)

# Save 'Id' for submission
test_ids = test_df['Id']
test_df = test_df.drop( ['Id'], axis=1 )


# In[41]:


print('random forest classifier...')
rfc_model = RandomForestClassifier(
    n_estimators=1000,
    n_jobs=-1
)
rfc_model.fit( train_df, train_label )
# rfc_pred = rfc_model.predict( test_df )

print('xgboost classifier...')
xgb_model = xgb.XGBClassifier(
    gamma=0.025,
    learning_rate=0.35,
    max_depth=9,
    n_estimators=1000,
    objective='multi:softmax',
    n_jobs=4
)
xgb_model.fit( train_df, train_label )
# xgb_pred = xgb_model.predict( test_df )

print('lgbm classifier...')
lgb_model = lgb.LGBMClassifier(
    learning_rate=0.2,
    max_depth=-1,
    n_estimators=1000,
    objective='multiclass',
    n_jobs=8,
    verbose=0
)
lgb_model.fit( train_df, train_label )
# lgb_pred = lgb_model.predict( test_df )


# In[45]:


rfc_pred_proba = rfc_model.predict_proba( test_df )
xgb_pred_proba = xgb_model.predict_proba( test_df )
lgb_pred_proba = lgb_model.predict_proba( test_df )


# In[46]:


blended_pred_proba = ( rfc_pred_proba + xgb_pred_proba + lgb_pred_proba ) / 3.0


# In[78]:


# display(blended_pred_proba.shape, blended_pred_proba)

def get_argmax( x_array ):
    return np.argmax( x_array )
get_argmax_vect = np.vectorize(get_argmax)

y_pred_max_proba = np.array( [get_argmax(x_row)+1 for x_row in blended_pred_proba] )


# In[84]:


submission = pd.DataFrame(
    y_pred_max_proba, index=test_ids, columns=['Cover_Type']
)
submission.to_csv('submission_baseline_rfc_lgb_xgb.csv', index_label='Id')


# In[ ]:


# Kaggle public leaderboard score (score - classification accuracy): 0.77121


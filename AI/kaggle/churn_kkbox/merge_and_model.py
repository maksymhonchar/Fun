#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# https://lightgbm.readthedocs.io/en/latest/Parameters.html


# In[13]:


import gc
gc.collect()


# In[9]:


# Load libraries

import pickle

import pandas as pd

from sklearn.model_selection import train_test_split

import lightgbm as lgb


# In[3]:


# Load train dataset

TRAIN_FILEPATH = 'data/train_v2.csv'
train_df = pd.read_csv(TRAIN_FILEPATH, header=0)

MEMBERS_FILEPATH = 'members_refined.pd.dataframe.pcl'
TRANSACTIONS_FILEPATH = 'transactions_refined.pd.dataframe.pcl'
USERLOGS_FILEPATH = 'user_logs_refined_final.pd.dataframe.pcl'


# In[4]:


# Merge the train with other tables

def _ugly_merge( fact_dataset_df ):
    # members
    print('merging members...')
    members_df = pickle.load( open(MEMBERS_FILEPATH, 'rb') )
    merged_df = pd.merge(
        left=fact_dataset_df, right=members_df,
        how='left',
        on=['msno']
    )
    del members_df
    # transactions
    print('merging transactions...')
    transactions_df = pickle.load( open(TRANSACTIONS_FILEPATH, 'rb') )
    merged_df = pd.merge(
        left=merged_df, right=transactions_df,
        how='left',
        on=['msno']
    )
    del transactions_df
    # userlogs
    print('merging userlogs...')
    userlogs_df = pickle.load( open(USERLOGS_FILEPATH, 'rb') )
    merged_df = pd.merge(
        left=merged_df, right=userlogs_df,
        how='left',
        on=['msno']
    )
    del userlogs_df    
    
    return merged_df


# In[5]:


merged_train = _ugly_merge( train_df )


# In[6]:


# Replace NaN values

display(merged_train.shape)

display(merged_train.isnull().sum())


# In[7]:


merged_train = merged_train.fillna( -1 )


# In[10]:


# Prepare X and y from train set

train_ids = merged_train['msno']
train_labels = merged_train['is_churn']
train_data = merged_train.drop( ['msno', 'is_churn'], axis=1 )

X_tr, X_val, y_tr, y_val = train_test_split(
    train_data, train_labels,
    test_size=0.35
)


# In[11]:


del merged_train


# In[15]:


X_tr_lgb = lgb.Dataset( X_tr, label=y_tr )
X_val_lgb = lgb.Dataset( X_val, label=y_val )

lgb_model_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'learning_rate': 0.25,
    'max_depth': 25,
    'num_leaves': 1500,
    
    'verbose': 1,
    
    'device': 'gpu',
    'n_jobs': 4  # number of real CPU cores
}


# In[18]:


lgb_model = lgb.train(
    lgb_model_params,
    train_set=X_tr_lgb,
    num_boost_round=500,
    valid_sets=[X_val_lgb]
)


# In[ ]:


TEST_FILEPATH = 'data/sample_submission_v2.csv'
test_df = pd.read_csv(TEST_FILEPATH, header=0)


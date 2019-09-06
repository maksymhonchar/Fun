#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Competition link:
# https://www.kaggle.com/c/web-traffic-time-series-forecasting/data


# In[2]:


# cool top1 guy vid: https://www.youtube.com/watch?v=0E1_vt_Z9HY
# cool yandex vid: https://www.youtube.com/watch?v=u3CBW_59pZA


# In[3]:


# Load libraries

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd


# In[4]:


# Load datasets

TRAIN_FILEPATH = 'data/train_1.csv'
TEST_FILEPATH = 'data/key_1.csv'

raw_train_df = pd.read_csv(TRAIN_FILEPATH, header=0)
raw_test_df = pd.read_csv(TEST_FILEPATH, header=0)


# In[5]:


raw_train_df.info()

raw_test_df.info()


# In[10]:


# Review duplicates

display(
    raw_train_df['Page'].duplicated().sum(),
    raw_test_df['Page'].duplicated().sum()
)


# In[11]:


# NaN values - only "visits on date X" features hold empty values

display(
    raw_train_df.isnull().sum()
)

# For now, replace NaN with "-1" value
# Don't replace with 0s - information about missing value might be useful later
VALUE_TO_REPLACE_NANS = -1
raw_train_df = raw_train_df.fillna( VALUE_TO_REPLACE_NANS )


# In[12]:


# Uppivot date features in train test set

date_features_names = raw_train_df.iloc[:, 1:].columns.values  # 1st feature is 'Page'

raw_train_df = pd.melt(
    raw_train_df,
    id_vars=['Page'],
    value_vars=date_features_names,
    var_name='Date',
    value_name='Visits'
)  # 5298 mb


# In[13]:


# Downcast floats to integers to save memory

raw_train_df['Visits'] = pd.to_numeric(
    raw_train_df['Visits'], downcast='integer'
)  # 4994 mb

# raw_train_df['Visits'] = raw_train_df['Visits'].astype( int )

# for col_name in raw_train_df.columns.values[1:]:  # omit 'Page' feature
#     raw_train_df[col_name] = pd.to_numeric(
#         raw_train_df[col_name], downcast='integer'
#     )


# In[18]:


# 'Page' feature: extract "name", "project", "access", "agent"
def split_page_feature( dataset_df ):
    splitted_df = dataset_df['Page'].str.rsplit( pat='_', n=3, expand=True )
    splitted_df.columns = [ 'name', 'project', 'access', 'agent' ]
    
    splitted_df['language'] = splitted_df['project'].str[:2]
    
    return splitted_df.join(dataset_df)


# In[ ]:


raw_train_df = split_page_feature( raw_train_df )

raw_test_df = split_page_feature( raw_test_df )


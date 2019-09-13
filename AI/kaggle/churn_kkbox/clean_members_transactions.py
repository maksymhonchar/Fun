#!/usr/bin/env python
# coding: utf-8

# In[14]:


# Load libraries

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd


# In[2]:


# Work on 'members' dataset

MEMBERS_FILEPATH = 'data/members_v3.csv'
members = pd.read_csv(MEMBERS_FILEPATH, header=0)


# In[3]:


members.info()

display(members.head())

display(members.isnull().sum())


# In[4]:


# Cast registration_init_time to datetime

members['registration_init_time'] = pd.to_datetime(
    members['registration_init_time'], format='%Y%m%d'
)

# Day should be 'relative' to some 0-coordinate
min_date = members['registration_init_time'].min()
members['registration_init_time'] -= min_date
members['registration_init_time'] = members['registration_init_time'].dt.days


# In[5]:


# Fix 'gender' feature

members['gender'] = members['gender'].fillna('NoGender')

# Encode the genders
members['gender'] = members['gender'].map({
    'NoGender': 1, 'male': 2, 'female':3 
})


# In[6]:


# Bin "registered_via" feature values
members['registered_via'].replace(
    [1, 2, 5, 6, 8, 10, 11, 13, 14, 16, 17, 18, 19, -1],
    1,
    inplace = True
)


# In[7]:


# Drop redundant features

members = members.drop( ['city', 'bd'], axis=1 )


# In[9]:


display(members.head())


# In[10]:


# Work on 'transactions' dataset

TRANSACTIONS_FILEPATH = 'data/transactions_v2.csv'
transactions = pd.read_csv(TRANSACTIONS_FILEPATH, header=0)


# In[11]:


display(transactions.shape)
display(transactions.sample())
transactions.info()


# In[12]:


# Convert xxx_date columns to datetime format

transactions['transaction_date'] = pd.to_datetime(
    transactions['transaction_date'], format='%Y%m%d'
)

transactions['membership_expire_date'] = pd.to_datetime(
    transactions['membership_expire_date'], format='%Y%m%d'
)


# In[13]:


# How many $ has the customer paid per day?

transactions['paid_per_day'] = transactions['actual_amount_paid'] / transactions['payment_plan_days']


# In[22]:


# Remove highly correlated features

sns.heatmap(
    transactions.corr(),
    annot=True
)
plt.autoscale()
plt.show()

# payment_plan_days - candidate


# In[23]:


transactions = transactions.drop( ['payment_plan_days', 'membership_expire_date', 'transaction_date'], axis=1 )


# In[26]:


transactions.head()


# In[25]:


import pickle

pickle.dump( members, open('members_refined.pd.dataframe.pcl', 'wb') )

pickle.dump( transactions, open('transactions_refined.pd.dataframe.pcl', 'wb') )


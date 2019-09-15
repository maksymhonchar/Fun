#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 1st place xgboost "paper": https://arxiv.org/pdf/1802.03396.pdf


# In[1]:


# Load libs

import pickle

import pandas as pd


# In[3]:


 def overview_df( dataset_df ):
    display(dataset_df.shape)
    display(dataset_df.sample())
    dataset_df.info()


# In[4]:


# Work on 'User logs' table

USER_LOGS_FILEPATH = 'data/user_logs_v2.csv'
user_logs = pd.read_csv(USER_LOGS_FILEPATH, header=0)

# overview_df(user_logs)


# In[5]:


# Convert seconds to hours

SECONDS_IN_HOUR = 3600

user_logs['total_hours'] = user_logs['total_secs'] / SECONDS_IN_HOUR


# In[6]:


# Temporary total number of songs listened / skipped
# Usage: calculate rate of num_xxx songs

user_logs['c_total_songs'] = user_logs['num_25'] + user_logs['num_50'] + user_logs['num_75'] + user_logs['num_985'] + user_logs['num_100']


# In[7]:


# Proportion of songs listened / skipped for each msno

user_logs['p_num_25'] = user_logs['num_25'] / user_logs['c_total_songs']
user_logs['p_num_50'] = user_logs['num_50'] / user_logs['c_total_songs']
user_logs['p_num_75'] = user_logs['num_75'] / user_logs['c_total_songs']
user_logs['p_num_985'] = user_logs['num_985'] / user_logs['c_total_songs']
user_logs['p_num_100'] = user_logs['num_100'] / user_logs['c_total_songs']


# In[8]:


# Convert date to datetime format

user_logs['date'] = pd.to_datetime( user_logs['date'], format='%Y%m%d' )

# Extract day number out of date column

user_logs['day_num'] = user_logs['date'].dt.day


# In[9]:


# Group user logs by msno

userlogs_groupby_msno = user_logs.groupby(by='msno')


# In[21]:


# Calculate features for each msno group

msno_params_mapping = {}  # {k:v} == msno:{params}

unique_msno_list = user_logs['msno'].unique()

for idx, msno in enumerate( unique_msno_list ):
    selected_group = userlogs_groupby_msno.get_group(msno)
    
    params = {}
    params['n_days'] = selected_group['day_num'].unique().shape[0]
    params['sum_day'] = sum( selected_group['day_num'] )
    params['last_day'] = max( selected_group['day_num'] )
    params['mean_total_secs'] = selected_group['total_secs'].mean()
    params['mean_total_songs'] = selected_group['c_total_songs'].mean()
    params['mean_num_25'] = selected_group['num_25'].mean()
    params['mean_num_50'] = selected_group['num_50'].mean()
    params['mean_num_75'] = selected_group['num_75'].mean()
    params['mean_num_985'] = selected_group['num_985'].mean()
    params['mean_num_100'] = selected_group['num_100'].mean()
    params['mean_p_num_25'] = selected_group['p_num_25'].mean()
    params['mean_p_num_50'] = selected_group['p_num_50'].mean()
    params['mean_p_num_75'] = selected_group['p_num_75'].mean()
    params['mean_p_num_985'] = selected_group['p_num_985'].mean()
    params['mean_p_num_100'] = selected_group['p_num_100'].mean()
    
    msno_params_mapping[msno] = params
    
    del selected_group  # +- 200 mb for each get_group
    del params
    
    # dbg
    if idx % 100000 == 0:
        print('{0} / {1}'.format(idx, len(unique_msno_list)))


# In[22]:


# Save params dictionary to hdd

pickle.dump( msno_params_mapping, open('msno_params_mapping.dict.pcl', 'wb') )


# In[52]:


# Load params to user_logs df

features_to_add = msno_params_mapping['EGcbTofOSOkMmQyN1NMLxHEXJ1yV3t/JdhGwQ9wXjnI='].keys()

for feature_name in features_to_add:    
    user_logs[feature_name] = user_logs['msno'].apply(
        lambda msno: msno_params_mapping[msno][feature_name]
    )
    print('feature "{0}" - done'.format(feature_name))
    
del msno_params_mapping


# In[55]:


# Save refined user_logs to hdd

pickle.dump( user_logs, open('user_logs_refined_backup.pd.dataframe.pcl', 'wb') )


# In[2]:


# Load and remove redundant features

user_logs = pickle.load( open('user_logs_refined.pd.dataframe.pcl', 'rb') )


# In[4]:


user_logs.dtypes


# In[5]:


columns_to_drop = [
    'date',
    'num_25', 'num_50', 'num_75', 'num_985', 'num_100',
    'total_secs',
    'c_total_songs',
    'p_num_25', 'p_num_50', 'p_num_75', 'p_num_985', 'p_num_100',
    'day_num'
]

user_logs = user_logs.drop( columns_to_drop, axis=1 )


# In[9]:


import sys
sys.getsizeof(user_logs)


# In[10]:


pickle.dump( user_logs, open('user_logs_refined_final.pd.dataframe.pcl', 'wb') )


# In[2]:


# Final check
import pickle
user_logs = pickle.load( open('user_logs_refined_final.pd.dataframe.pcl', 'rb') )
display(user_logs.info())
display(user_logs.head())


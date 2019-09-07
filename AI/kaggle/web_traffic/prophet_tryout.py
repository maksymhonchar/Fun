#!/usr/bin/env python
# coding: utf-8

# In[9]:


import gc
gc.collect()


# In[2]:


# FBProphet:
    # faster than LSTM
    # note all the patterns from previous data
    
# https://facebook.github.io/prophet/
# https://facebook.github.io/prophet/docs/quick_start.html

# Prophet: multiple plots bug in Jupyter: https://github.com/facebook/prophet/issues/124


# In[44]:


# Load libraries

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

import pandas as pd

from fbprophet import Prophet


# In[4]:


# Load datasets

train_df = pd.read_csv('data/train_1.csv')


# In[5]:


# Drop features not related to dates

train_df = train_df.drop( ['Page'], axis=1 )


# In[7]:


# Fix null values

# display(
#     train_df.isnull().sum().describe()
# )

# Even if Prophet doesn't care about NaNs - fill them up anyways

# todo: fill with 1s, 0s - compare results

NAN_VALUES_REPLACEMENT = 1
train_df = train_df.fillna( NAN_VALUES_REPLACEMENT )


# In[8]:


# Downcast every 'views amount' to integer format

train_df = train_df.astype(int)


# In[14]:


# reshaped df to use in Prophet

# Note: names "ds" and "y" are mandatory

reshaped_page = pd.DataFrame({
    'ds': train_df.T.index.values,
    'y': train_df.values[0]
})


# In[58]:


# Try out prediction for 1st row

prophet_model = Prophet()

prophet_model.fit( reshaped_page )

pred_df = prophet_model.make_future_dataframe( periods=100 )
pred = prophet_model.predict( pred_df )

prophet_model.plot(pred);

# Note: there are bunch of outliers -> remove them


# In[54]:


# Try out prediction v2 WITHOUT outliers

reshaped_page = pd.DataFrame({
    'ds': train_df.T.index.values,
    'y': train_df.values[0]
})

percentile_5 = reshaped_page['y'].quantile(0.05)
percentile_95 = reshaped_page['y'].quantile(0.95)

below_5_percentile_filter = reshaped_page['y'] <= percentile_5
above_95_percentile_filter = reshaped_page['y'] >= percentile_95

display(
    reshaped_page[ below_5_percentile_filter | above_95_percentile_filter ].shape,
    reshaped_page.shape
)  # 64 dates out of 550 dates are outliers;

# Replace outliers with None value - Prophet will ignore these values

indices_to_replace = reshaped_page[
    below_5_percentile_filter | above_95_percentile_filter ].index

reshaped_page.loc[ indices_to_replace, 'y' ] = None


# In[57]:


# Try out Prophet prediction again - without outliers

prophet_model = Prophet()

prophet_model.fit( reshaped_page )

pred_df = prophet_model.make_future_dataframe( periods=100 )
pred = prophet_model.predict( pred_df )

prophet_model.plot(pred);


# In[59]:


# next -> predict in a similar way for each page


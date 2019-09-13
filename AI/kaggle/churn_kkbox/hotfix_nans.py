#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns

import pickle
import numpy as np
import pandas as pd


# In[4]:


MEMBERS_FILEPATH = 'members_refined.pd.dataframe.pcl'
members_df = pickle.load( open(MEMBERS_FILEPATH, 'rb') )


# In[27]:


display(
    members_df.isnull().sum()
)


# In[30]:


sns.distplot(
    members_df.gender
)
plt.show()


# In[37]:


pickle.dump( members_df, open('members_refined.pd.dataframe.pcl', 'wb') )


# In[ ]:





# In[6]:


TRANSACTIONS_FILEPATH = 'transactions_refined.pd.dataframe.pcl'
transactions_df = pickle.load( open(TRANSACTIONS_FILEPATH, 'rb') )


# In[7]:


display(
    transactions_df.isnull().sum()
)


# In[25]:


# display(transactions_df.paid_per_day.min())
# display(transactions_df.paid_per_day.max())

# display(
#     transactions_df[transactions_df['paid_per_day'] == np.inf]
# )

mean_paidperday_value = transactions_df[transactions_df['paid_per_day'] != np.inf]['paid_per_day'].mean()
display(mean_paidperday_value)

transactions_df['paid_per_day'] = transactions_df['paid_per_day'].replace(
    np.inf,
    mean_paidperday_value
)

# Fix NaNs also
transactions_df['paid_per_day'] = transactions_df['paid_per_day'].fillna( mean_paidperday_value )


# In[36]:


sns.distplot(
    transactions_df.paid_per_day
)
plt.show()


# In[38]:


pickle.dump( transactions_df, open('transactions_refined.pd.dataframe.pcl', 'wb') )


# In[ ]:





# In[3]:


USERLOGS_FILEPATH = 'user_logs_refined_final.pd.dataframe.pcl'
userlogs = pickle.load( open(USERLOGS_FILEPATH, 'rb') )


# In[5]:


display(
    userlogs.isnull().sum()
)


# In[14]:


sns.distplot(
    userlogs.mean_p_num_100
)
plt.show()


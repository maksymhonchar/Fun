#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gc
gc.collect()


# In[2]:


# Load libraries

get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd


# In[3]:


# Load data

raw_train_data = pd.read_csv('data/train_1.csv')


# In[1]:


# Fill NaN values with 0s
# note: this might be a harmful move

# todo: replace by group? season?
# todo: new feature: season? 

NAN_VALUES_REPLACEMENT = 1
train_data = raw_train_data.fillna( NAN_VALUES_REPLACEMENT )


# In[5]:


# Overview the data

# display( raw_train_data.sample() )
# display( raw_train_data.shape )
# display( raw_train_data.isnull().sum() )

# raw_train_data.info()


# In[6]:


# Reformat 'Page' feature into a single 'Language' feature

train_splitted_df = train_data['Page'].str.rsplit(
    pat='_', n=3, expand=True
)
train_splitted_df.columns = [ 'name', 'project', 'access', 'agent' ]

display(train_splitted_df['project'].value_counts())

languages = train_splitted_df['project'].str[:2]

display(languages.value_counts())

# reformatting:
train_data['Page'] = languages


# In[7]:


language_groupby = train_data.groupby( by='Page' )

lang_sums = dict()

for name, group in language_groupby:
    lang_sums[name] = group.iloc[:, 1:].sum( axis=0 )
    lang_sums[name] /= group.index.size    


# In[10]:


x_values = np.linspace( 0, len(lang_sums['en']), len(lang_sums['en']) )

fig = plt.figure( figsize=[25, 7] )

for key in lang_sums.keys():
    plt.plot(x_values, lang_sums[key], label=key)
    
plt.ylabel('no of Views')
plt.xlabel('day number')
plt.legend()
plt.show()


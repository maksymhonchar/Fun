#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gc
gc.collect()


# In[2]:


# Load libs

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd

import numpy as np


# In[3]:


# Load datasets

bidders = pd.read_csv('data/train.csv', header=0)
bids = pd.read_csv('data/bids.csv', header=0)


# In[4]:


# Training and Test sets to work with later / create features / encode etc

train = bidders.copy()
test = pd.read_csv('data/test.csv', header=0)


# In[5]:


# Overview time of the bids

sns.distplot(
    bids['time'], bins=100
)
plt.show()

display(
    pd.cut(bids['time'], 3).value_counts()
)


# In[6]:


# Identify one of the three periods shown on the graph above
# Use np.log for better readability

time_binning_cut = pd.cut( bids['time'], 3 )
bids['time_bin'] = time_binning_cut

bids['time_bin'] = bids['time_bin'].apply(
    lambda x: '{0:.4f}_{1:.4f}'.format( np.log(x.left), np.log(x.right) )
)

# Add feature to train and test: 'time_bin'
bids_grouped_by_bidders = bids.groupby(by='bidder_id')

bidder_no_auctions_time_bin_replacement = bids['time_bin'].value_counts().index[1]

def apply_time_bin( bidder_id ):
    try:
        bidder_group = bids_grouped_by_bidders.get_group(bidder_id)
        time_bin_value = bidder_group['time_bin'].values[0]
        return time_bin_value
    except:
        return bidder_no_auctions_time_bin_replacement

print('train: creating time_bin...')
train['time_bin'] = train['bidder_id'].apply(
    lambda x: apply_time_bin( x )
)
print('test: creating time_bin')
test['time_bin'] = test['bidder_id'].apply(
    lambda x: apply_time_bin( x )
)


# In[7]:


# First and last auction for each bidder
# For bidders which didn't took part in any auction: replace with random time.

bidder_no_auctions_min_time_replacement = bids['time'].median()
bidder_no_auctions_max_time_replacement = bidder_no_auctions_min_time_replacement


def apply_min_time( bidder_id ):
    try:
        bidder_group = bids_grouped_by_bidders.get_group( bidder_id )
        min_time_value = bidder_group['time'].min()
        return min_time_value
    except KeyError:
        return bidder_no_auctions_min_time_replacement

    
def apply_max_time( bidder_id ):
    try:
        bidder_group = bids_grouped_by_bidders.get_group( bidder_id )
        max_time_value = bidder_group['time'].max()
        return max_time_value
    except KeyError:
        return bidder_no_auctions_max_time_replacement
    
print('train: creating min_time...')
train['min_time'] = train['bidder_id'].apply(
    lambda x: apply_min_time( x )
)
print('test: creating min_time...')
test['min_time'] = test['bidder_id'].apply(
    lambda x: apply_min_time( x )
)
print('train: creating max_time...')
train['max_time'] = train['bidder_id'].apply(
    lambda x: apply_max_time( x )
)
print('test: creating max_time...')
test['max_time'] = test['bidder_id'].apply(
    lambda x: apply_max_time( x )
)


# In[8]:


# Amount of bids done by single bidder

single_bidder_auctions = bids[ ['bidder_id', 'auction'] ].groupby(by='bidder_id')
single_bidder_auctions_cnt = single_bidder_auctions.count()['auction']

print('train: creating c_bids...')
train['c_bids'] = train['bidder_id'].apply(
    lambda x: single_bidder_auctions_cnt.get(x, 0)  # 0 if no auctions found
)
print('test: creating c_bids...')
test['c_bids'] = test['bidder_id'].apply(
    lambda x: single_bidder_auctions_cnt.get(x, 0)
)


# In[9]:


# 29 rows without any activity in train set
# all of these rows were made by humans
display(train[train['c_bids'] == 0].shape)

# 70 rows without any activity in test set
# also humans?
display(test[test['c_bids'] == 0].shape)


# In[20]:


# Last auction bid

def create_auction_max_auc_time_mapping():
    single_auction_bids = bids.groupby( by='auction' )
    auction_max_auc_time_mapping = dict()
    for name, group in single_auction_bids:
        auction_max_auc_time_mapping[name] = group['time'].max()
    return auction_max_auc_time_mapping

train_auction_max_auc_time_mapping = create_auction_max_auc_time_mapping()

bids['max_auction_time'] = bids['auction'].apply(
    lambda x: train_auction_max_auc_time_mapping[x]
)


# In[68]:


# Time between bid time and ending of auction
# Return np.log( mean_value )

def apply_mean_aucend_tdiff(bidder_id):
    try:
        bidder_group = bids_grouped_by_bidders.get_group( bidder_id )
        aucend_tdiff = bidder_group['max_auction_time'] - bidder_group['time']
        mean_aucend_tdiff = np.mean(aucend_tdiff)
        return np.log( mean_aucend_tdiff )
    except KeyError:
        return 0

print('train: creating mean_aucend...')
train['mean_aucend'] = train['bidder_id'].apply(
    lambda x: apply_mean_aucend_tdiff( x )
)
print('test: creating mean_aucend...')
test['mean_aucend'] = test['bidder_id'].apply(
    lambda x: apply_mean_aucend_tdiff( x )
)

# Test contains 2 -inf values (missing certain auctions in bids df)
# Replace them with "0" value
test['mean_aucend'] = test['mean_aucend'].replace(-np.inf, 0)

# Replace 0s with average value to fix distribution
train_mean_aucend_mean_value = train['mean_aucend'].mean()
train['mean_aucend'] = train['mean_aucend'].replace(0, train_mean_aucend_mean_value)

test_mean_aucend_mean_value = test['mean_aucend'].mean()
test['mean_aucend'] = test['mean_aucend'].replace(0, test_mean_aucend_mean_value)


# In[69]:


sns.distplot(
    train['mean_aucend'], bins=100
)
plt.show()

sns.distplot(
    test['mean_aucend'], bins=100
)
plt.show()


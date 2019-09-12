#!/usr/bin/env python
# coding: utf-8

# In[1]:


# NOTE: please, look at sns.distplot for each of train/test set! 
# It is really compelling to see humans vs bots distributions


# In[2]:


import gc
gc.collect()


# In[32]:


# Load libs

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report


# In[4]:


# Load datasets

bidders = pd.read_csv('data/train.csv', header=0)
bids = pd.read_csv('data/bids.csv', header=0)


# In[39]:


# Training and Test sets to work with later / create features / encode etc

train = bidders.copy()
test = pd.read_csv('data/test.csv', header=0)


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


# For each bid, calculate time passed from last bid.

bids_grouped_by_auction = bids.groupby( by='auction' )
bidid_bidresponse_mapping = {}

display('dummy: creating mapping bid_id to prev_time')
for name, group in bids_grouped_by_auction:
    group_sorted = group.sort_values(by='time')
    group_sorted['prev_time'] = group_sorted['time'].shift(1)
    time_col_idx = group_sorted.columns.get_loc('time')
    next_data_col_idx = group_sorted.columns.get_loc('prev_time')
    group_sorted.iloc[0, next_data_col_idx] = group_sorted.iloc[0, time_col_idx]
    group_sorted['bid_response'] = group_sorted['time'] - group_sorted['prev_time']
    for i in group_sorted.index.values:
        bidid_bidresponse_mapping[i] = group_sorted.at[i, 'bid_response']


# In[10]:


bids['bid_response'] = bids['bid_id'].apply(
    lambda bidid: np.log1p( bidid_bidresponse_mapping[bidid] )
)

# del bidid_bidresponse_mapping


# In[11]:


# Apply mean, min and max response for each bidder

def apply_aggfunc_bid_response( bidder_id, aggfunc, except_return=0 ):
    try:
        selected_bids = bids_grouped_by_bidders.get_group( bidder_id )
        aggfunc_log_bin_response = aggfunc( selected_bids['bid_response'] )
        return aggfunc_log_bin_response
    except KeyError:
        return except_return


print('train: mean bid_response')
train['mean_log_bid_response'] = train['bidder_id'].apply(
    lambda x: apply_aggfunc_bid_response( x, pd.Series.mean )
)
print('test: mean bid_response')
test['mean_log_bid_response'] = test['bidder_id'].apply(
    lambda x: apply_aggfunc_bid_response( x, pd.Series.mean )
)

print('train: min bid_response')
train['min_log_bid_response'] = train['bidder_id'].apply(
    lambda x: apply_aggfunc_bid_response( x, pd.Series.min )
)
print('test: min bid_response')
test['min_log_bid_response'] = test['bidder_id'].apply(
    lambda x: apply_aggfunc_bid_response( x, pd.Series.min )
)

print('train: max bid_response')
train['max_log_bid_response'] = train['bidder_id'].apply(
    lambda x: apply_aggfunc_bid_response( x, pd.Series.max )
)
print('test: max bid_response')
test['max_log_bid_response'] = test['bidder_id'].apply(
    lambda x: apply_aggfunc_bid_response( x, pd.Series.max )
)


# In[12]:


# Mark bids that were 'winning': last ones

bidid_iswinningbid_mapping = {}

print('dummy: creating bidid_iswinningbid mapping')
for name, group in bids_grouped_by_auction:
    group_sorted = group.sort_values(by='time')
    winning_bid = group_sorted.iloc[-1, :]
    bidid_iswinningbid_mapping[ winning_bid['bid_id'] ] = 1


# In[13]:


# Apply "bid_id -> flag_winnin_bid" mapping to bids

bids['flag_is_winning_bid'] = bids['bid_id'].apply(
    lambda x: bidid_iswinningbid_mapping.get( x, 0 )
)

# del bidid_iswinningbid_mapping


# In[14]:


# Total amount of wins in auctions

def apply_cwonauct( bidder_id ):
    try:
        selected_group = bids_grouped_by_bidders.get_group( bidder_id )
        return selected_group['flag_is_winning_bid'].sum()
    except KeyError:
        return 0
    

print('train: c_wonauct...')
train['c_wonauct'] = train['bidder_id'].apply(
    lambda x: apply_cwonauct( x )
)
print('test: c_wonauct...')
test['c_wonauct'] = test['bidder_id'].apply(
    lambda x: apply_cwonauct( x )
)


# In[15]:


# Amount of unique countries, IPs, URLs for each bidder

def apply_distinct_colname_count( bidder_id, col_name ):
    try:
        selected_group = bids_grouped_by_bidders.get_group( bidder_id )
        distinct_colname_count = len( selected_group[col_name].unique() )
        return distinct_colname_count
    except:
        return 0
    
    
print('train: distinct countries...')
train['c_unq_countries'] = train['bidder_id'].apply(
    lambda x: apply_distinct_colname_count(x, 'country')
)
print('test: distinct countries...')
test['c_unq_countries'] = test['bidder_id'].apply(
    lambda x: apply_distinct_colname_count(x, 'country')
)

print('train: distinct IPs...')
train['c_unq_ips'] = train['bidder_id'].apply(
    lambda x: apply_distinct_colname_count(x, 'ip')
)
print('test: distinct IPs...')
test['c_unq_ips'] = test['bidder_id'].apply(
    lambda x: apply_distinct_colname_count(x, 'ip')
)

print('train: distinct URLs...')
train['c_unq_urls'] = train['bidder_id'].apply(
    lambda x: apply_distinct_colname_count(x, 'url')
)
print('test: distinct URLs...')
test['c_unq_urls'] = test['bidder_id'].apply(
    lambda x: apply_distinct_colname_count(x, 'url')
)

print('train: distinct devices...')
train['c_unq_devices'] = train['bidder_id'].apply(
    lambda x: apply_distinct_colname_count(x, 'device')
)
print('test: distinct devices...')
test['c_unq_devices'] = test['bidder_id'].apply(
    lambda x: apply_distinct_colname_count(x, 'device')
)


# In[16]:


# Drop redundant features

train = train.drop(['payment_account', 'address'], axis=1)
test = test.drop(['payment_account', 'address'], axis=1)

# display(train.head())
# display(test.head())


# In[17]:


# Prepare the X and y to fit classifier

train_ids = train['bidder_id']
test_ids = test['bidder_id']

train_labels = train['outcome']

train = train.drop(['bidder_id', 'outcome'], axis=1)
test = test.drop(['bidder_id'], axis=1)

# Encode time_bin column
timebin_lbl_encoder = LabelEncoder()
train['time_bin'] = timebin_lbl_encoder.fit_transform( train['time_bin'] )
test['time_bin'] = timebin_lbl_encoder.transform( test['time_bin'] )

# Log-transform max_time and min_time columns: other time-related columns are log-transformed too
train['min_time'] = np.log1p( train['min_time'] )
train['max_time'] = np.log1p( train['max_time'] )
test['min_time'] = np.log1p( test['min_time'] )
test['max_time'] = np.log1p( test['max_time'] )


# In[18]:


train.head()


# In[19]:


test.head()


# In[27]:


# Build the model

X_train = train
y_train = train_labels

rfc_model = RandomForestClassifier(
    n_estimators=1000, max_depth=25, min_samples_leaf=3,
    n_jobs=-1, verbose=1
)

X_tr, X_val, y_tr, y_val = train_test_split( X_train, y_train, test_size=0.35, shuffle=True )

rfc_model.fit( X_tr, y_tr )


# In[34]:


# report

y_pred = rfc_model.predict( X_val )

print( classification_report( y_val, y_pred ) )

# .score
display( rfc_model.score(X_val, y_val) )


# In[24]:


# CV
cv_scores = cross_val_score(rfc_model, X_train, y_train, cv=10)

# display(rfc_model.score())
display(cv_scores.mean())


# In[30]:


display(
    pd.DataFrame(
        {'col_name': X_train.columns.values, 'feat_imp': rfc_model.feature_importances_}
    ).sort_values(by='feat_imp')
)


# In[38]:


# Create a submission

test_pred = rfc_model.predict_proba( test )

submission = pd.DataFrame({
    'bidder_id': test_ids,
    'prediction': test_pred[:, 1]
})

submission.to_csv('submission.csv', index=False)


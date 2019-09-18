#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gc
gc.collect()


# In[2]:


# Load libraries

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

import pandas as pd


# In[3]:


# Load bureau-related datasets

BUREAU_FILEPATH = 'data/bureau.csv'
bureau_df = pd.read_csv( BUREAU_FILEPATH, header=0 )

BUREAUBALANCE_FILEPATH = 'data/bureau_balance.csv'
bureaubalance_df = pd.read_csv( BUREAUBALANCE_FILEPATH, header=0 )

# Merge bureau data alltogether
bureau_mrgd_df = pd.merge(
    left=bureau_df, right=bureaubalance_df,
    how='left',
    on='SK_ID_BUREAU'
)
bureau_mrgd_df = bureau_mrgd_df.drop( ['SK_ID_BUREAU'], axis=1 )

del bureau_df
del bureaubalance_df
gc.collect()


# In[4]:


display(
#     bureau_df.shape, bureaubalance_df.shape,
    bureau_mrgd_df.shape
)  # skip 2,178,110 rows using left join (25,121,815 rows left)


# In[5]:


def overview_df( df ):
    display( df.shape )
    display( df.head(1) )
    display( 'np.nan values:', df.isnull().sum() )
    display( 'duplicates:', df.duplicated().sum() )
    
    df.info()


# In[6]:


# overview_df( bureau_mrgd_df )


# In[7]:


# STATUS
# Status of Credit Bureau loan during the month

# Values:
# C means closed
# X means status unknown
# 0 means no DPD (days past due)
# 1 means maximal did during month between 1-30
# 2 means DPD 31-60
# 3 means DPD 61-90
# 4 means DPD 91-120
# 5 means DPD 120+ or sold or written off

# About DPD:
# Past due means overdue.
# Typically, a bill is past due if the borrower is 30 days past
# the payment deadline.
# src: https://investinganswers.com/dictionary/p/past-due

display( bureau_mrgd_df.groupby( by='STATUS' ).size() )

# 1. Create 'dpd_level' feature: amount of days past due, based on bureau_balance 'STATUS' feature
# Let 'C' and 'V' describe 'dpd=0' value: assume closed and unknown contracts have dpd=0
# For now, leave rows with bureau_balance=np.nan as having np.nan value - deal with them later

def apply_dpd_level( status_value ):
    if pd.isnull( status_value ):
        NAN_STATUS_CONTRACTS_DPD = np.nan
        return NAN_STATUS_CONTRACTS_DPD
    elif status_value in ('X', 'C'):
        CLOSED_UNKNOWN_STATUS_CONTRACTS_DPD = 0
        return CLOSED_UNKNOWN_STATUS_CONTRACTS_DPD
    else:
        return int( status_value )


bureau_mrgd_df['dpd_level'] = bureau_mrgd_df['STATUS'].apply(
    lambda status_value: apply_dpd_level( status_value )
)

# 2. Flag 'Unknown' STATUS feature values (where STATUS='X')
# This will ensure we could distinguish STATUS=C and STATUS=X values in dpd_level feature (these 2 categories have dpd_level=0)

UNKOWN_STATUS_INDICATOR = 'X'
FLAG_UNKNOWN_STATUS = 1
FLAG_NOT_UNKNOWN_STATUS = 0
bureau_mrgd_df['flg_status_x'] = bureau_mrgd_df['STATUS'].apply(
    lambda x: FLAG_UNKNOWN_STATUS if x==UNKOWN_STATUS_INDICATOR else FLAG_NOT_UNKNOWN_STATUS
)


# In[26]:


# Apply aggregating functions (sum, mean, max, std) to dpd-related features
# for every unique borrower.

borrowers_groupby = bureau_mrgd_df.groupby( by='SK_ID_CURR' )
unique_borrowers_ids = bureau_mrgd_df['SK_ID_CURR'].unique()

# 1. Apply sum, mean, max, std to 'dpd_level' feature
def apply_skidcur_dpdlevel_agg( sk_id_cur, agg_func ):
    try:
        selected_group = borrowers_groupby.get_group( sk_id_cur )
        agg_value = 
        display(selected_group)
    except KeyError:
        return np.nan
    
display( borrowers_groupby.get_group( 215354 ) )


# In[8]:


# MONTHS_BALANCE
# Month of balance relative to application date
# -1 means the freshest balance date; time only relative to the application

# 1. Inverse (* -1) the values - easier to explain
bureau_mrgd_df['MONTHS_BALANCE'] = bureau_mrgd_df['MONTHS_BALANCE'] * ( -1 )

bureau_mrgd_df['MONTHS_BALANCE'].hist( bins=20 )
plt.show()

# 2. Create bins explaining 1 out of 4 quartiles of MONTHS_BALANCE feature values.
# NaN values will have np.nan values
QUARTILES_AMNT = 4
bureau_mrgd_df['mon_balance_quart'] = pd.qcut( bureau_mrgd_df['MONTHS_BALANCE'], QUARTILES_AMNT )
# Convert 'mon_balance_quart' to 'str' format
bureau_mrgd_df['mon_balance_quart'] = bureau_mrgd_df['mon_balance_quart'].apply(
    lambda x: '{0}_{1}'.format( x.left, x.right )
)

# 3. Create bins explaining custom ranges of MONTHS_BALANCE:
# <7 days, 7-14 days, 14-30 days, 30-60 days, >60 days
def apply_mon_balance_custom_ranges( mon_balance_value ):
    if pd.isnull( mon_balance_value ):
        return np.nan
    if mon_balance_value < 7:  # 1 week
        return '1_week_mon_balance'
    elif mon_balance_value < 14:  # 1-2 weeks
        return '2_weeks_mon_balance'
    elif mon_balance_value < 28:  # 2-4 weeks
        return '4_weeks_mon_balance'
    elif mon_balance_value < 60:  # 1-2 months
        return '2_months_mon_balance'
    else:  # other: >2 months
        return '>2_months_mon_balance'

bureau_mrgd_df['mon_balance_custom_ranges'] = bureau_mrgd_df['MONTHS_BALANCE'].apply(
    lambda mon_balance_value: apply_mon_balance_custom_ranges( mon_balance_value )
)


# In[11]:


bureau_mrgd_df['mon_balance_quart'].value_counts()


# In[ ]:





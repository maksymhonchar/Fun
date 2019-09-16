#!/usr/bin/env python
# coding: utf-8

# In[1]:


# competition src: https://www.kaggle.com/c/forest-cover-type-prediction
# data src: https://www.kaggle.com/c/3936/download-all


# In[2]:


import gc
gc.collect()


# In[160]:


# Load libraries

import re

import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd


# In[4]:


# Load train and test dataset

TRAIN_FILEPATH = 'data/train.csv'
train_df = pd.read_csv( TRAIN_FILEPATH, header=0 )

TEST_FILEPATH = 'data/test.csv'
test_df = pd.read_csv( TEST_FILEPATH, header=0 )


# In[5]:


def overview_df( df ):
    display( df.sample() )
    display( df.shape )
    display( df.isnull().sum() )
    display( df.duplicated().sum() )
    df.info()


# In[96]:


overview_df( train_df )

overview_df( test_df )


# In[7]:


# Transform multiple Wilderness_Areax or Soil_Typex-like features a into single categorical feature

def merge_onehot( dataset_df, col_name_no_x ):
    """Convert col_nameX features to single feature
    X means some integer value.
    Doesn't work with multiple calls - returns 0s for all col_name_no_x.
    """
    dataset_df_cpy = dataset_df.copy()
    # 1. Identify columns
    all_df_columns = dataset_df_cpy.columns.values
    re_pattern_compiled = re.compile( "^{0}(\d+)$".format( col_name_no_x ) )
    matched_columns = list(filter( re_pattern_compiled.match, all_df_columns ))
    # 2. Change columns: multiply by 'x' value
    for matched_column in matched_columns:
        col_name_x_value = re_pattern_compiled.match(matched_column).groups()[0]
        dataset_df_cpy[matched_column] *= int( col_name_x_value ) 
    # 3. Merge col_namex columns into single col_name column
    dataset_df_cpy[col_name_no_x] = 0
    for matched_column in matched_columns:
        dataset_df_cpy[col_name_no_x] += dataset_df_cpy[matched_column]
    # 4. Drop col_namex columns
    dataset_df_cpy = dataset_df_cpy.drop( matched_columns, axis=1 )
    
    return dataset_df_cpy
    
def _ugly_merge_wildernessarea_soiltype_traintest( train_or_test_df ):
    train_or_test_df = merge_onehot( train_or_test_df, col_name_no_x='Wilderness_Area' )
    train_or_test_df = merge_onehot( train_or_test_df, col_name_no_x='Soil_Type' )
    return train_or_test_df


# In[8]:


train_df = _ugly_merge_wildernessarea_soiltype_traintest( train_df )

test_df = _ugly_merge_wildernessarea_soiltype_traintest( test_df )


# In[9]:


# Remove redundant (for eda) features

train_target = train_df['Cover_Type']
train_df = train_df.drop( ['Id', 'Cover_Type'], axis=1 )

test_df = test_df.drop( ['Id'], axis=1 )


# In[74]:


def overview_distribution( df, col_name, ax,                          title_text='', n_bins=100, display_kde=False,
                          **hist_kwargs):
    df[col_name].hist( bins=n_bins, ax=ax, label='density=True freq', density=True, **hist_kwargs )
    if display_kde:
        df[col_name].plot.kde( ax=ax, color='red', label='kde' )
    ax.set_xlabel(col_name)
    ax.set_ylabel('freq, normalized')
    ax.grid(True)
    ax.legend()
    ax.set_title('{0} distribution {1}'.format(col_name, title_text))
    

def overview_traintest_distributions( train_df, test_df, col_name, axes ):
    sns.distplot( train_df[col_name], ax=axes[0], label='train' )
    sns.distplot( test_df[col_name], ax=axes[0], label='test' )
    axes[0].set_title('{0} distribution | TRAIN, TEST'.format(col_name))
    axes[0].legend()
    overview_distribution( train_df, col_name, ax=axes[1], title_text='| TRAIN' )
    overview_distribution( test_df, col_name, ax=axes[2], title_text='| TEST' )
    
    
def quick_traintest_distr_overview( train_df, test_df, col_name, figsize_tuple=(15, 5) ):
    fig, axes = plt.subplots( 1, 3, figsize=figsize_tuple)
    overview_traintest_distributions( train_df, test_df, col_name, axes=axes )


# In[84]:


# Overview distributions

for col_name in train_df.columns.values:
    quick_traintest_distr_overview( train_df, test_df, col_name )


# In[109]:


# Overview distributions relative to target type

train_df['Cover_Type'] = train_target

for col_name in train_df.columns.values[:-1]:
    for cover_type in sorted( train_df['Cover_Type'].unique() ):        
        sns.distplot(
            train_df[ train_df['Cover_Type'] == cover_type ][col_name],
            label=cover_type,
            bins=50
        )
    plt.legend()
    plt.show()


# In[95]:


# Overview correlation in train / test sets

def overview_pearson_corr( df ):
    fig = plt.figure( figsize=(15, 15) )
    sns.heatmap(
        df.corr(),
        annot=True
    )
    plt.autoscale()
    plt.show()

train_df['Cover_Type'] = train_target
overview_pearson_corr( train_df )

overview_pearson_corr( test_df )


# In[126]:


# Skewnewss

def overview_bad_kurtosis_skewness( df, df_str_descr ):
    # kurtosis
    print('{0}: Features, where kurtosis not as in normal univariate distribution:'.format(df_str_descr))
    df_kurt = df.kurtosis()
    display(
        df_kurt[ (df_kurt < -2) | (df_kurt > 2) ]
    )
    # skew
    print('{0}: Features, where skew not as in normal univariate distribution:'.format(df_str_descr))
    df_skewness = df.skew()
    display(
        df_skewness[ (df_skewness < -1) | (df_skewness > 1) ]
    )


overview_bad_kurtosis_skewness( train_df, 'TRAIN' )
overview_bad_kurtosis_skewness( test_df, 'TEST' )


# In[162]:


# Overview scatter plots relative to target type

def overview_scatter_x_y_color_covertype( df, x, y, hue='Cover_Type', size='Elevation' ):    
    fig = plt.figure(figsize=(15, 15))
    sns.scatterplot(
        x=x, y=y,
        hue=hue,
        size=size,
        data=df,
    )
    plt.show()

overview_scatter_x_y_color_covertype(
    train_df, 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology'
)

overview_scatter_x_y_color_covertype(
    train_df, 'Hillshade_3pm', 'Hillshade_9am'
)

overview_scatter_x_y_color_covertype(
    train_df, 'Aspect', 'Hillshade_Noon'
)

overview_scatter_x_y_color_covertype(
    train_df, 'Slope', 'Hillshade_Noon'
)

overview_scatter_x_y_color_covertype(
    train_df, 'Elevation', 'Horizontal_Distance_To_Roadways'
)


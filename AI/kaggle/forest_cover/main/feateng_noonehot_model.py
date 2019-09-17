#!/usr/bin/env python
# coding: utf-8

# In[17]:


# Import libraries

import re

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, classification_report

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb
import lightgbm as lgb


# In[2]:


# Load datasets

TRAIN_FILEPATH = 'data/train.csv'
train_df = pd.read_csv( TRAIN_FILEPATH, header=0 )
display(train_df.shape)

TEST_FILEPATH = 'data/test.csv'
test_df = pd.read_csv( TEST_FILEPATH, header=0 )
display(test_df.shape)


# In[3]:


# Fix values

# 1. As we can see, some of 'Hillshade_3pm' values = 0 -> fix this
train_df['Hillshade_9am'].hist(bins=200, label='9am')
train_df['Hillshade_3pm'].hist(bins=200, label='3pm')
train_df['Hillshade_Noon'].hist(bins=200, label='Noon')
plt.title('train before')
plt.legend()
plt.show()

display(
    'train before:',
    train_df[ train_df['Hillshade_3pm'] < 5 ]['Hillshade_3pm'].value_counts()
)

test_df['Hillshade_9am'].hist(bins=200, label='9am')
test_df['Hillshade_3pm'].hist(bins=200, label='3pm')
test_df['Hillshade_Noon'].hist(bins=200, label='Noon')
plt.title('test before')
plt.legend()
plt.show()

display(
    'test before',
    test_df[ test_df['Hillshade_3pm'] < 5 ]['Hillshade_3pm'].value_counts()
)

# Use data from test set (500k rows), forget about train set (15k)
hillshade_pred_useful_columns = [
    col_name for col_name
    in test_df.columns.values
    if col_name not in ['Hillshade_3pm', 'Id', 'Cover_Type']
]
train_topredict_rows = train_df[ train_df['Hillshade_3pm'] == 0 ][hillshade_pred_useful_columns]
test_topredict_rows = test_df[ test_df['Hillshade_3pm'] == 0 ][hillshade_pred_useful_columns]

totrain_rows_X = test_df[ test_df['Hillshade_3pm'] != 0 ][hillshade_pred_useful_columns]
totrain_rows_y = test_df[ test_df['Hillshade_3pm'] != 0 ]['Hillshade_3pm']

predictor = KNeighborsRegressor(n_neighbors=10)  # todo: estimate accuracy
predictor.fit( totrain_rows_X, totrain_rows_y )

train_df.loc[ train_topredict_rows.index.values, 'Hillshade_3pm' ] = np.around( predictor.predict(train_topredict_rows) )
test_df.loc[ test_topredict_rows.index.values, 'Hillshade_3pm' ] = np.around( predictor.predict(test_topredict_rows) )

# 'after' result :
plt.title('train after')
train_df['Hillshade_9am'].hist(bins=200, label='9am')
train_df['Hillshade_3pm'].hist(bins=200, label='3pm')
train_df['Hillshade_Noon'].hist(bins=200, label='Noon')
plt.legend()
plt.show()

display(
    'train after:',
    train_df[ train_df['Hillshade_3pm'] < 5 ]['Hillshade_3pm'].value_counts()
)

plt.title('test after')
test_df['Hillshade_9am'].hist(bins=200, label='9am')
test_df['Hillshade_3pm'].hist(bins=200, label='3pm')
test_df['Hillshade_Noon'].hist(bins=200, label='Noon')
plt.legend()
plt.show()

display(
    'test after',
    test_df[ test_df['Hillshade_3pm'] < 5 ]['Hillshade_3pm'].value_counts()
)


# In[4]:


# Work with concatenated feautures

traintest_df = pd.concat( [train_df, test_df], sort=False, ignore_index=True )


# In[5]:


# Combine Soil_TypeX and Wilderness_AreaX into single feature

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


# train_df = _ugly_merge_wildernessarea_soiltype_traintest( train_df )
# test_df = _ugly_merge_wildernessarea_soiltype_traintest( test_df )

# display(train_df.shape, test_df.shape)

traintest_df = _ugly_merge_wildernessarea_soiltype_traintest( traintest_df )


# In[6]:


# Feature engineering

def add_soil_family_inplace( df ):
    # src: https://www.kaggle.com/c/forest-cover-type-prediction/data
    # (soiltypeX, soiltypeY, ...): family_name_str
    soiltype_family_mapping = {
        (2, 4): 1, # 'ratake',
        (10, 11, 13, 32, 33): 2, # 'catamount',
        (21, 22, 23, 24, 25, 27, 28): 3, # 'leighcan',
        (38, 39, 40): 4, # 'moran'
#         (... all other IDs ...): 'other_type'
    }
    df['soil_family'] = 5  # 'other_type'
    for i in df.index:
        soil_type_value = df.at[i, 'Soil_Type']
        for key in soiltype_family_mapping.keys():
            if soil_type_value in key:
                df.at[i, 'soil_family'] = soiltype_family_mapping[key]    


def add_soil_complex_inplace( df ):
    # (soiltypeX, soiltypeY, ...): complex_name_str
    soiltype_complex_mapping = {
        (1, 3, 4, 5, 6, 10, 27, 28, 33): 1, # 'rock_outcrop',
        (11, 12, 34, 40): 2, # 'rock_land',
        (20, 23): 3, # 'typic_cryaquolls',
        (26, 31): 4, # 'catamaount_families',
        (29, 30): 5, # 'legault_family',
        (32, 39): 6, # 'leighcan_family',
    }
    df['soil_complex'] = 7  # 'other_type'
    for i in df.index:
        soil_type_value = df.at[i, 'Soil_Type']
        for key in soiltype_complex_mapping.keys():
            if soil_type_value in key:
                df.at[i, 'soil_complex'] = soiltype_complex_mapping[key]    


def add_soil_stonetype_inplace( df ):
    # (soiltypeX, soiltypeY, ...): stonetype_name_str
    soiltype_stonetype_mapping = {
        (1, 2, 6, 9, 12, 18, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40): 1, # 'stony',
        (3, 4, 5, 10, 11, 13, ): 2, # 'rubbly'
    }
    df['soil_stonetype'] = 3 # 'other_type'
    for i in df.index:
        soil_type_value = df.at[i, 'Soil_Type']
        for key in soiltype_stonetype_mapping.keys():
            if soil_type_value in key:
                df.at[i, 'soil_stonetype'] = soiltype_stonetype_mapping[key]    


def main_feature_engineering( df ):
    df_cpy = df.copy()
    
    # Median hillshade index [0-255]
    df_cpy['median_hillshade_idx'] = df_cpy[['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']].agg('median', axis='columns')
    # Mean hillshade index [0-255]
    df_cpy['median_hillshade_idx'] = df_cpy[['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']].agg('mean', axis='columns')
    # Distance to hydrology (using HorizontalDistance and VerticalDistance to hydrology) - pythagoras theorem
    df_cpy['hydrology_distance'] = np.sqrt( df_cpy['Horizontal_Distance_To_Hydrology']**2 + df_cpy['Vertical_Distance_To_Hydrology']**2 )
    # Aspect binning: 20 intervals
    ASPECT_BINS_CNT = 20
    df_cpy['aspect_bin'] = pd.cut( df_cpy['Aspect'], bins=ASPECT_BINS_CNT )
    df_cpy['aspect_bin'] = df_cpy['aspect_bin'].apply(
        lambda interval: '{0}_{1}'.format(interval.left, interval.right)
    )
    # Polynomial features: dependance of 9am->noon->3pm->9am
    df_cpy['9am_noon_dep'] = df_cpy['Hillshade_9am'] * df_cpy['Hillshade_Noon']
    df_cpy['noon_3pm_dep'] = df_cpy['Hillshade_Noon'] * df_cpy['Hillshade_3pm']
    df_cpy['3pm_9am_dep'] = df_cpy['Hillshade_3pm'] * df_cpy['Hillshade_9am']
    # Cosine of slope: relationships between hillshade and other features
    df_cpy['slope_cosine'] = np.cos( df_cpy['Slope'] )
    # Log-transform 'Elevation' feature
    df_cpy['Elevation'] = np.log1p( df_cpy['Elevation'] )
    
    
    return df_cpy


def apply_feature_engineering( df ):
    print('soil family...')
    add_soil_family_inplace( df )
    print('soil complex...')
    add_soil_complex_inplace( df )
    print('soil stonetype...')
    add_soil_stonetype_inplace( df )
    print('main feature engineering...')
    df = main_feature_engineering( df )
    return df


# train_df = apply_feature_engineering( train_df )
# test_df = apply_feature_engineering( test_df )

traintest_df = apply_feature_engineering( traintest_df )


# In[7]:


# 29th - x3
# 15 - only 3 values

# display(
#     traintest_df['Soil_Type'].value_counts()
# )
# traintest_df.shape


# In[8]:


# Convert 'aspect_bin' to numerical format

aspectbin_lblencoder = LabelEncoder()
traintest_df['aspect_bin'] = aspectbin_lblencoder.fit_transform( traintest_df['aspect_bin'] )


# In[9]:


# Split traintest df for model validation

train = traintest_df.iloc[:train_df.shape[0], :]
test = traintest_df.iloc[train_df.shape[0]:, :]

# remove redundant columns
train_labels = train['Cover_Type']
train = train.drop( ['Id', 'Cover_Type'], axis=1 )
test = test.drop( ['Id'], axis=1 )

# train-validation split
VALIDATION_SIZE = 0.3
X_tr, X_val, y_tr, y_val = train_test_split(
    train, train_labels,
    test_size=VALIDATION_SIZE,
    shuffle=True
)

display(X_tr.shape, X_val.shape)


# In[10]:


# 1. Try out lgbm

lgb_model = lgb.LGBMClassifier(
    learning_rate=0.25,
    max_depth=-1,
    n_estimators=1000,
    objective='multiclass',
    n_jobs=8,
    verbose=1
)
lgb_model.fit( X_tr, y_tr )
lgb_y_val_pred = lgb_model.predict( X_val )

display( accuracy_score(y_val, lgb_y_val_pred) )
print( classification_report(y_val, lgb_y_val_pred) )
display( multilabel_confusion_matrix(y_val, lgb_y_val_pred) )


# In[11]:


display(
    pd.DataFrame({
        'feature_name': X_tr.columns.values,
        'feature_imp': lgb_model.feature_importances_
    }).sort_values(by='feature_imp', ascending=False)
)


# In[12]:


# 2. Try out xgb

xgb_model = xgb.XGBClassifier(
    gamma=0.03,
    learning_rate=0.2,
    max_depth=5,
    n_estimators=1000,
    objective='multi:softmax',
    n_jobs=4
)
xgb_model.fit( X_tr, y_tr )
xgb_y_val_pred = xgb_model.predict( X_val )

display( accuracy_score(y_val, xgb_y_val_pred) )
print( classification_report(y_val, xgb_y_val_pred) )
display( multilabel_confusion_matrix(y_val, xgb_y_val_pred) )


# In[13]:


display(
    pd.DataFrame({
        'feature_name': X_tr.columns.values,
        'feature_imp': xgb_model.feature_importances_
    }).sort_values(by='feature_imp', ascending=False)
)


# In[14]:


# 3. Try out rfc

rfc_model = RandomForestClassifier(
    n_estimators=1000,
    n_jobs=-1
)
rfc_model.fit( X_tr, y_tr )
rfc_y_val_pred = rfc_model.predict( X_val )

display( accuracy_score(y_val, rfc_y_val_pred) )
print( classification_report(y_val, rfc_y_val_pred) )
display( multilabel_confusion_matrix(y_val, rfc_y_val_pred) )


# In[15]:


display(
    pd.DataFrame({
        'feature_name': X_tr.columns.values,
        'feature_imp': rfc_model.feature_importances_
    }).sort_values(by='feature_imp', ascending=False)
)


# In[16]:


# Try to increase models performance by fixing skewness

# tofix_skew_col_names = []
# for col_name in traintest_df:
#     skew_value = traintest_df[col_name].skew()
#     if not -1 < skew_value < 1:
#         tofix_skew_col_names.append( col_name )      
        
# Cannot apply boxcox1p for all columns
# for col_name in tofix_skew_col_names:
#     try:
#         boxcox_norm = boxcox_normmax( trainteset_df[col_name] + 1 )
#         display( col_name, boxcox1p(traintest_df[col_name], boxcox_norm).skew() )
#     except:
#         display('cannot apply boxcox for {0}'.format(col_name))

# cant do that


# In[24]:


# Find best hyperparameters for RFC, XGB, LGB classifiers

# 1. RFC
rfc_model = RandomForestClassifier( )
rfc_param_grid = {
    'n_estimators': [100, 250, 500, 1000],
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 4, 5, 10, None],
    'min_samples_split': np.linspace(0.1, 1, 5),
    'max_features': [2, 5, 'auto'],
    'n_jobs': [4],
}
rfc_grid_search = GridSearchCV(
    estimator=rfc_model,
    param_grid=rfc_param_grid,
    cv=5,
    verbose=2, iid=False, n_jobs=4
)
rfc_grid_search.fit( X_tr, y_tr )
display( rfc_grid_search.best_params_ )


# In[37]:


# 2. LGBM

lgb_model = lgb.LGBMClassifier(
    objective='multiclass',
    n_jobs=4, verbose=0
)
lgb_grid_params = {
    'learning_rate': [0.2, 0.25, 0.3],
    'num_leaves': [ int(val) for val in np.linspace(5, 25, 3) ],
    'max_depth': [-1, 5, 15, 25],
    'n_estimators': [100, 250, 500, 1000],
    'min_split_gain': [0.0, 0.05, 0.1]
}
lgb_grid_search = GridSearchCV(
    estimator=lgb_model,
    param_grid=lgb_grid_params,
    cv=2,
    verbose=2, iid=False, n_jobs=4
)
lgb_grid_search.fit( X_tr, y_tr )
display( lgb_grid_search.best_params_ )


# In[ ]:


# 3. XGB

xgb_model = xgb.XGBClassifier(
    objective='multi:softmax',
    n_jobs=4, verbosity=0
)
xgb_grid_params = {
    'gamma': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 25],
    'n_estimators': [250, 500],
}
xgb_grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=xgb_grid_params,
    cv=2,
    verbose=2, iid=False, n_jobs=4
)
xgb_grid_search.fit( X_tr, y_tr )
display( xgb_grid_search.best_params_ )


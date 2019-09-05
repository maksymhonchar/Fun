#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Kmeans + XGB classifier


# In[2]:


# Load libraries

import pickle

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from sklearn.decomposition import PCA
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.cluster import KMeans

import lightgbm as lgb
import xgboost as xgb


# In[3]:


# Load datasets

raw_train_df = pd.read_csv('data/train.csv', parse_dates=['Dates'])
raw_test_df = pd.read_csv('data/test.csv', parse_dates=['Dates'])


# In[4]:


# display(raw_train_df.shape, raw_test_df.shape)

# display(raw_train_df.sample(5), raw_test_df.sample(5))

# display(raw_train_df.info(), raw_test_df.info())


# In[5]:


# Remove duplicates from train set

display( raw_train_df.duplicated().sum() )

raw_train_df = raw_train_df.drop_duplicates()


# In[6]:


# Fix invalid coordinates

INCORRECT_X_COORD_VALUE = -120.5
INCORRECT_Y_COORD_VALUE = 90.0


def _ugly_fix_coords_inplace(dataset_df, 
                             x_coord_col_name='X', y_coord_col_name='Y'):
    mean_x_value = dataset_df[x_coord_col_name].mean()
    mean_y_value = dataset_df[y_coord_col_name].mean()
    
    dataset_df[x_coord_col_name] = dataset_df[x_coord_col_name].apply(
        lambda x: mean_x_value if x == INCORRECT_X_COORD_VALUE else x 
    )
    
    dataset_df[y_coord_col_name] = dataset_df[y_coord_col_name].apply(
        lambda y: mean_y_value if y == INCORRECT_Y_COORD_VALUE else y 
    )


# In[7]:


display( raw_train_df[ raw_train_df['Y'] == 90.0 ].shape )
_ugly_fix_coords_inplace( raw_train_df )

display( raw_test_df[ raw_test_df['Y'] == 90.0 ].shape )
_ugly_fix_coords_inplace( raw_test_df )


# In[8]:


# Feature engineering

def add_new_features_inplace( dataset_df ):
    dataset_df['tmp_date'] = pd.to_datetime( dataset_df['Dates'].dt.date )
    
    # calendar
    dataset_df['days_from_0'] = dataset_df['tmp_date'] - dataset_df['tmp_date'].min()
    dataset_df['days_from_0'] = dataset_df['days_from_0'].apply(
        lambda x: x.days
    )
    
    dataset_df['Day'] = dataset_df['Dates'].dt.day
    dataset_df['DayOfWeek'] = dataset_df['Dates'].dt.weekday
    dataset_df['Month'] = dataset_df['Dates'].dt.month
    dataset_df['Year'] = dataset_df['Dates'].dt.year
    
    # clock
    dataset_df['Hour'] = dataset_df['Dates'].dt.hour
    dataset_df['Minute'] = dataset_df['Dates'].dt.minute
    
    # coordinates
    dataset_df['X_Y'] = dataset_df['X'] - dataset_df['Y']
    dataset_df['XY'] = dataset_df['X'] + dataset_df['Y']
    
    # address
    dataset_df['Intersection'] = dataset_df['Address'].str.contains( '/', case=False )
    dataset_df['Intersection'] = dataset_df['Intersection'].apply(
        lambda x: int( x )
    )


# In[9]:


add_new_features_inplace( raw_train_df )

add_new_features_inplace( raw_test_df )


# In[10]:


# display(raw_train_df.shape, raw_test_df.shape)

# display(raw_train_df.sample(5), raw_test_df.sample(5))

# display(raw_train_df.info(), raw_test_df.info())


# In[11]:


# Remove unnecessary features

train_df = raw_train_df.drop( ['Dates', 'Descript', 'Resolution', 'Address', 'tmp_date'], axis=1 )

test_df = raw_test_df.drop( ['Id', 'Dates', 'Address', 'tmp_date'], axis=1 )


# In[12]:


# Separate label feature

y_category = train_df.loc[:, 'Category']
train_df = train_df.drop( ['Category'], axis=1 )


# In[13]:


display(
    y_category.head(),
    train_df.head(),
    test_df.head()
)


# In[14]:


# Transform X,Y coordinates to 0,0 special coordinates using PCA

def transform_x_y_coords_pca_inplace( dataset_df ):
    pca_decompositor = PCA( n_components=2 )
    new_coordinates = pca_decompositor.fit_transform( dataset_df[ ['X', 'Y'] ] )
    
    dataset_df['pca_coord1'] = new_coordinates[:, 0]
    dataset_df['pca_coord2'] = new_coordinates[:, 1]


# In[15]:


transform_x_y_coords_pca_inplace( train_df )

transform_x_y_coords_pca_inplace( test_df )


# In[16]:


display(
    y_category.head(),
    train_df.head(),
    test_df.head()
)


# In[17]:


# Use k-means clustering algorithm to detect cluster 

# no of clusters - as seen in tableau, there are 6 districts with > 20k crimes (data from train.csv)
# => no of clusters = 6

def find_clusters_kmeans_inplace( dataset_df ):
    kmeans = KMeans( n_clusters=6, n_jobs=-1 )
    found_clusters = kmeans.fit_predict( dataset_df[ ['X', 'Y'] ] )
    
    dataset_df['geo_cluster'] = found_clusters    


# In[18]:


find_clusters_kmeans_inplace( train_df )

find_clusters_kmeans_inplace( test_df )


# In[19]:


display(
    y_category.head(),
    train_df.head(),
    test_df.head()
)


# In[20]:


# Prepare training data to feed into model: encoding

# Labeling

distr_enc = LabelEncoder()
train_df['PdDistrict'] = distr_enc.fit_transform( train_df['PdDistrict'] )
test_df['PdDistrict'] = distr_enc.transform( test_df['PdDistrict'] )

cat_enc = LabelEncoder()   # note: later could be used for inverse_transform when creating a submission
y_category = cat_enc.fit_transform( y_category )


# In[21]:


# One-hot encoding

col_names_to_encode = ['DayOfWeek', 'geo_cluster']

train_df = pd.get_dummies( train_df, columns=col_names_to_encode, drop_first=True )
test_df = pd.get_dummies( test_df, columns=col_names_to_encode, drop_first=True )


# In[22]:


display(
    y_category[:5],
    train_df.head(),
    test_df.head()
)


# In[23]:


# Train model on prepared training data

TRAIN_SIZE = 0.8

X_tr, X_val, y_tr, y_val = train_test_split(
    train_df, y_category,
    train_size=TRAIN_SIZE, shuffle=True,
    random_state=42
)


# In[24]:


display(X_tr.shape, X_val.shape)

display(
    pd.unique(y_tr).shape,  # should be 39
    pd.unique(y_val).shape  # should be 39
)


# In[54]:


# # Model 1. lgb

# lgb_train_data = lgb.Dataset(
#     X_tr, label=y_tr,
#     categorical_feature=['PdDistrict', ]
# )

# classes_amount = pd.unique(y_category).shape[0]  # should be 39
# display(classes_amount)

# lgb_params = {
#     'objective':'multiclass',
#     'num_class': classes_amount,
#     'max_delta_step': 0.87,
#     'min_data_in_leaf': 18,
#     'num_leaves': 41,
#     'learning_rate': 0.30,
#     'max_bin': 500,
#     'n_jobs': 4
# }

# lgb_model = lgb.train( lgb_params, lgb_train_data, num_boost_round=150)

# lgb_y_pred = lgb_model.predict( X_val )

# display(
#     log_loss(
#         y_val,
#         lgb_y_pred
#     )    
# )  # 2.236597257698194


# In[62]:


# # Model 2. lgb v2 (decr min_data_in_leaf to 5)

# lgb_train_data = lgb.Dataset(
#     X_tr, label=y_tr,
#     categorical_feature=['PdDistrict', ]
# )

# classes_amount = pd.unique(y_category).shape[0]  # should be 39
# display(classes_amount)

# lgb_params = {
#     'objective':'multiclass',
#     'num_class': classes_amount,
#     'max_delta_step': 0.87,
#     'min_data_in_leaf': 5,
#     'num_leaves': 41,
#     'learning_rate': 0.30,
#     'max_bin': 500,
#     'n_jobs': 4  # 4 real cores
# }

# lgb_model = lgb.train( lgb_params, lgb_train_data, num_boost_round=150)

# lgb_y_pred = lgb_model.predict( X_val )

# display(
#     log_loss(
#         y_val,
#         lgb_y_pred
#     )    
# )  # 2.2375462167091418


# In[65]:


# # Model 3. lgb v3

# lgb_train_data = lgb.Dataset(
#     X_tr, label=y_tr,
#     categorical_feature=['PdDistrict', ]
# )

# classes_amount = pd.unique(y_category).shape[0]  # should be 39
# display(classes_amount)

# lgb_params = {
#     'objective':'multiclass',
#     'num_class': classes_amount,
#     'max_delta_step': 0.80,
#     'min_data_in_leaf': 10,
#     'num_leaves': 30,
#     'learning_rate': 0.15,
#     'max_bin': 350,
#     'n_jobs': 4  # 4 real cores
# }

# lgb_model = lgb.train( lgb_params, lgb_train_data, num_boost_round=100)

# lgb_y_pred = lgb_model.predict( X_val )

# display(
#     log_loss(
#         y_val,
#         lgb_y_pred
#     )    
# )  # 2.2622893093038856


# In[68]:


# # Model 4. lgb v4: take lgb v1 and increase epochs to 200

# lgb_train_data = lgb.Dataset(
#     X_tr, label=y_tr,
#     categorical_feature=['PdDistrict', ]
# )

# classes_amount = pd.unique(y_category).shape[0]  # should be 39
# display(classes_amount)

# lgb_params = {
#     'objective':'multiclass',
#     'num_class': classes_amount,
#     'max_delta_step': 0.87,
#     'min_data_in_leaf': 18,
#     'num_leaves': 41,
#     'learning_rate': 0.30,
#     'max_bin': 500,
#     'n_jobs': 4
# }

# lgb_model = lgb.train( lgb_params, lgb_train_data, num_boost_round=200)

# lgb_y_pred = lgb_model.predict( X_val )

# display(
#     log_loss(
#         y_val,
#         lgb_y_pred
#     )    
# )  # 2.2356411724813006


# In[71]:


# # Model 5: lgb v5: take lgb v1 and decrease epochs to 80

# lgb_train_data = lgb.Dataset(
#     X_tr, label=y_tr,
#     categorical_feature=['PdDistrict', ]
# )

# classes_amount = pd.unique(y_category).shape[0]  # should be 39
# display(classes_amount)

# lgb_params = {
#     'objective':'multiclass',
#     'num_class': classes_amount,
#     'max_delta_step': 0.87,
#     'min_data_in_leaf': 18,
#     'num_leaves': 41,
#     'learning_rate': 0.30,
#     'max_bin': 500,
#     'n_jobs': 4
# }

# lgb_model = lgb.train( lgb_params, lgb_train_data, num_boost_round=80)

# lgb_y_pred = lgb_model.predict( X_val )

# display(
#     log_loss(
#         y_val,
#         lgb_y_pred
#     )    
# )  # 2.2450219208103714


# In[88]:


# # Model 6. xgboost

# xgb_train_data = xgb.DMatrix(X_tr, label=y_tr)
# xgb_val_data = xgb.DMatrix(X_val)

# xgb_params = {
#     'objective': 'multi:softprob',
#     'max_depth': 5,
#     'eval_metric': 'mlogloss',
#     'learning_rate': 0.07,
#     'num_class': 39,  # as earlier: 39 distinct categories in train.csv
#     'nthread': 8,
#     'num_parallel_tree': 4,
#     'tree_method': 'gpu_hist',
# }

# xgb_model = xgb.train( xgb_params, xgb_train_data, num_boost_round=50)

# xgb_y_pred = xgb_model.predict( xgb_val_data )

# display(
#     log_loss(
#         y_val,
#         xgb_y_pred
#     )    
# )  # 2.4139587918914276


# In[91]:


# # Model 7: xgboost v2: same as v1 but with decreased epochs

# xgb_train_data = xgb.DMatrix(X_tr, label=y_tr)
# xgb_val_data = xgb.DMatrix(X_val)

# xgb_params = {
#     'objective': 'multi:softprob',
#     'max_depth': 5,
#     'eval_metric': 'mlogloss',
#     'learning_rate': 0.07,
#     'num_class': 39,  # as earlier: 39 distinct categories in train.csv
#     'nthread': 8,
#     'num_parallel_tree': 4,
#     'tree_method': 'gpu_hist',
# }

# xgb_model = xgb.train( xgb_params, xgb_train_data, num_boost_round=30)

# xgb_y_pred = xgb_model.predict( xgb_val_data )

# display(
#     log_loss(
#         y_val,
#         xgb_y_pred
#     )    
# )  # 2.5418781892982816


# In[94]:


# # Model 8: xgboost v3: same as v1 but with increased epochs

# xgb_train_data = xgb.DMatrix(X_tr, label=y_tr)
# xgb_val_data = xgb.DMatrix(X_val)

# xgb_params = {
#     'objective': 'multi:softprob',
#     'max_depth': 5,
#     'eval_metric': 'mlogloss',
#     'learning_rate': 0.07,
#     'num_class': 39,  # as earlier: 39 distinct categories in train.csv
#     'nthread': 8,
#     'num_parallel_tree': 7,  # leave 1 for gpu ? сомнительно
#     'tree_method': 'gpu_hist',
# }

# xgb_model = xgb.train( xgb_params, xgb_train_data, num_boost_round=100)

# xgb_y_pred = xgb_model.predict( xgb_val_data )

# display(
#     log_loss(
#         y_val,
#         xgb_y_pred
#     )    
# )  # 2.321969730283627


# # Building models

# In[25]:


# Prepare data sets for xgb/lgb classifiers

# amount of distinct classes to predict
classes_amount = pd.unique( y_category ).shape[0]  # 39
display(classes_amount)

# lgb

lgb_train_data = lgb.Dataset(
    train_df, label=y_category,
    categorical_feature=['PdDistrict', ]
)
# lgb_test_data == test_df

# xgb

xgb_train_data = xgb.DMatrix(
    train_df, label=y_category
)

xgb_test_data = xgb.DMatrix(
    test_df
)


# In[26]:


# Method for quick submission

def create_submission_csv(csv_filename, prediction_proba):
    """Note: use external cat_enc LabelEncoder.
    Note: rename 'Id' by your hand in submission file
    """
    submission_df = pd.DataFrame(
        prediction_proba,
        columns = cat_enc.inverse_transform( np.linspace(0, 38, 39, dtype='int16') )
    )
    submission_df_rounded = submission_df.round(5)

    print('saving csv on hdd...')
    submission_df_rounded.to_csv(csv_filename)


# In[27]:


# Method for quick model saving / loading

def pickle_save_model(filename, model_obj):
    pickle.dump(model_obj, open(filename, 'wb'))
    
def pickle_load_model(filename):
    return pickle.load(open(filename, 'rb'))


# # Approach 1

# In[99]:


# # Approach 1: single best logloss model

# # Best model was lgbv4 with logloss score "2.2356" on validation data

# lgb_params = {
#     'objective':'multiclass',
#     'num_class': classes_amount,
#     'max_delta_step': 0.87,
#     'min_data_in_leaf': 18,
#     'num_leaves': 41,
#     'learning_rate': 0.30,
#     'max_bin': 500,
#     'n_jobs': 4
# }

# print('training...')
# lgb_model = lgb.train( lgb_params, lgb_train_data, num_boost_round=200)

# pickle_save_model('top1_lgb.model', lgb_model)

# print('predicting...')
# lgb_y_pred = lgb_model.predict( test_df )

# print('Approach 1: done')

# create_submission_csv('approach1_submission.csv')

# # Kaggle logloss result: 2.26630


# # Approach 2

# In[ ]:


# Approach 2: blend 2 best models (1 from xgb, 1 from lgb)

# Best models are lgb v4 and xgb v3

# Create & save top1 xgb model

xgb_params = {
    'objective': 'multi:softprob',
    'max_depth': 5,
    'eval_metric': 'mlogloss',
    'learning_rate': 0.07,
    'num_class': 39,  # as earlier: 39 distinct categories in train.csv
    'nthread': 8,
    'num_parallel_tree': 4,
    'tree_method': 'gpu_hist',
}

print('training...')
xgb_model = xgb.train( xgb_params, xgb_train_data, num_boost_round=100)

pickle_save_model('top1_xgb.model', lgb_model)

print('predicting xgb...')
xgb_y_pred = xgb_model.predict( xgb_test_data )

# Load top 1 lgb model
top1_lgb_model = pickle_load_model('top1_lgb.model')

# Make a prediction for top1 lgb model

print('predicting lgb...')
lgb_y_pred = lgb_model.predict( test_df )

# Make a final blended prediction

concat_y_pred = (lgb_y_pred + xgb_y_pred) / 2

create_submission_csv('approach1_submission.csv')

print('Approach 2: done')


# In[ ]:


# todo: one-hot for geo_cluster ?????

# todo: don't one-hot encode weekday

# todo: bin some continuous data (like create "seasons" feature and so on)


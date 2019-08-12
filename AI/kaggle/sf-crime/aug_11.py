#!/usr/bin/env python
# coding: utf-8

# In[1]:


# src: https://www.kaggle.com/c/sf-crime


# In[2]:


# cool yandex vid: https://www.youtube.com/watch?v=ttVhB8ET5M8
# cool paper: https://arxiv.org/pdf/1607.03626.pdf
# cool paper[2]: https://pdfs.semanticscholar.org/962c/019196d5901bd6cf4826cec1c075ab3de5ee.pdf


# In[3]:


# From kaggle 'Data Description' section:
    
# This dataset contains incidents derived from SFPD Crime
# Incident Reporting system.

# The data ranges from 1/1/2003 to 5/13/2015.

# The training set and test set rotate every week,
# meaning week 1,3,5,7... belong to test set,
# week 2,4,6,8 belong to training set. 

# Data fields
# Dates - timestamp of the crime incident
# Category - category of the crime incident (only in train.csv).
    # This is the target variable you are going to predict.TRwe
# Descript - detailed description of the crime incident (only in train.csv)
# DayOfWeek - the day of the week
# PdDistrict - name of the Police Department District
# Resolution - how the crime incident was resolved (only in train.csv)
# Address - the approximate street address of the crime incident 
# X - Longitude
# Y - Latitude


# In[4]:


# Load libraries

import IPython

import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = [10.0, 10.0]
import matplotlib.pyplot as plt

import gmplot

import numpy as np

import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

from scipy.stats import kstest, probplot

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier


# In[5]:


# Load the data

TRAIN_DF_PATH = 'data/train.csv'
TEST_DF_PATH = 'data/test.csv'

raw_train_df = pd.read_csv(TRAIN_DF_PATH, header=0)
raw_test_df = pd.read_csv(TEST_DF_PATH, header=0)

raw_concat_traintest_df = pd.concat([raw_train_df, raw_test_df], ignore_index=True, sort=False)


# In[6]:


def overview_df(dataset_df):
    # Elements in dataset
    display(dataset_df.sample(5))
    # Dataset shape
    display(dataset_df.shape)
    # Columns and dtypes
    display(dataset_df.dtypes)
    # .describe method
    display(dataset_df.describe(include='all').T)
    # Empty columns
    display(dataset_df.isnull().sum())


# In[7]:


overview_df(raw_train_df)

overview_df(raw_test_df)

overview_df(raw_concat_traintest_df)


# In[8]:


# Overview rows with unusual Longtitude and Latitude

# "Unusual" means incorrect latitude/longtitude range
# Latitudes range: [-90;+90]. Longtitudes range: [-180;+180].

# Note: all the rows have the same feature values: X=-120.5, Y=90.0.
# Note: there is same type of invalid rows in both train and test sets.

# Note: using overview_col_name_occurences_fulldf_subsetdf function: some "invalid" rows have 
    # valid X,Y coordinates in the training set.
# Note: using overview_col_name_occurences_fulldf_subsetdf, it is better to use concat df


def overview_invalid_long_lat(dataset_df):
    # Look for invalid rows
    invalid_long_rows = dataset_df[
        (dataset_df['X'] <= -180) | (dataset_df['X'] >= 0)
    ]
    invalid_lat_rows = dataset_df[
        (dataset_df['Y'] >= 90) | (dataset_df['Y'] <= 0)
    ]
    # Review amount of rows with invalid longtitude / latitude values
    display("Found longtitude invalid values: {0}".format(invalid_long_rows.shape))
    display("Found latitude invalid values: {0}".format(invalid_lat_rows.shape))


def overview_col_name_occurences_fulldf_subsetdf(full_dataset_df, subset_df, col_name):
    # Remove subset from the full dataset to omit using subset_df values.
    # It is expected that subset_df is in full_dataset_df.
    tosearch_df = full_dataset_df.drop(subset_df.index)
    # Iterate through subset_df and find which rows in full_dataset_df have the same value in col_name.
    for subset_col_name_val in subset_df[col_name].sort_values():
        occurences = tosearch_df[ tosearch_df[col_name] == subset_col_name_val ]
        if occurences.shape[0] != 0:
            display(
                'value "{0}" from subset has {1} occurences in full_dataset'.format(
                    subset_col_name_val, occurences.shape[0]
                )
            )
#             display(occurences)


# In[9]:


overview_invalid_long_lat(raw_train_df)
display('-')

overview_invalid_long_lat(raw_test_df)
display('-')

overview_invalid_long_lat(raw_concat_traintest_df)
display('-')

overview_col_name_occurences_fulldf_subsetdf(
    raw_train_df, raw_train_df[ raw_train_df['Y'] == 90.0 ], 'Address')
display('-')

overview_col_name_occurences_fulldf_subsetdf(
    raw_test_df, raw_test_df[ raw_test_df['Y'] == 90.0 ], 'Address')
display('-')

overview_col_name_occurences_fulldf_subsetdf(
    raw_concat_traintest_df, raw_concat_traintest_df[ raw_concat_traintest_df['Y'] == 90.0 ], 'Address')
display('-')


# In[10]:


# Plot concatenated coordinates
concat_no_invalid = raw_concat_traintest_df[ raw_concat_traintest_df['Y'] != 90.0 ]

fig = plt.figure(figsize=(10, 10))
plt.scatter(
    x=concat_no_invalid['X'], y=concat_no_invalid['Y'],
    s=1
)
plt.show()

# Plot coordinates separately from train/test sets
train_no_invalid = raw_train_df[ raw_train_df['Y'] != 90.0 ]
test_no_invalid = raw_test_df[ raw_test_df['Y'] != 90.0 ]
fig = plt.figure(figsize=(20, 20))
plt.scatter(
    x=train_no_invalid['X'], y=train_no_invalid['Y'],
    s=10, marker='v', alpha=0.2
)
plt.scatter(
    x=test_no_invalid['X'], y=test_no_invalid['Y'],
    s=10, marker='^', alpha=0.2
)
plt.show()


# In[11]:


# Generate geographical heatmaps
# Loading full train/test sets requires too much memory - plot subset of all coords

# Train set
train_lon, train_lat = train_no_invalid['X'], train_no_invalid['Y']

train_gmap = gmplot.GoogleMapPlotter(train_lat[0], train_lon[0], 12)
train_gmap.heatmap(train_lat[:50000], train_lon[:50000])
train_gmap.draw('train_50k_rows_heatmap.html')

train_gmap_html_iframe = IPython.display.IFrame(src='train_50k_rows_heatmap.html', width=800, height=400)
display(train_gmap_html_iframe)

# Test set
test_lon, test_lat = test_no_invalid['X'], test_no_invalid['Y']

test_gmap = gmplot.GoogleMapPlotter(train_lat[0], train_lon[0], 12)  # not test_*[0] to compare pics
test_gmap.heatmap(test_lat[:50000], test_lon[:50000])
test_gmap.draw('test_50k_rows_heatmap.html')

test_gmap_html_iframe = IPython.display.IFrame(src='test_50k_rows_heatmap.html', width=800, height=400)
display(test_gmap_html_iframe)


# In[12]:


# Note: in train and test datasets, X and Y coordinates are almost the same
fig, [ax_0, ax_1, ax_2, ax_3] = plt.subplots(1, 4, figsize=(20, 7))
train_no_invalid['X'].hist(ax=ax_0, bins=50)
test_no_invalid['X'].hist(ax=ax_1, bins=50)
train_no_invalid['Y'].hist(ax=ax_2, bins=50)
test_no_invalid['Y'].hist(ax=ax_3, bins=50)
plt.show()

# Note: X and Y are somehow "normally" distributed (kstat tells they are not), left skewed distribution.
fig, [ax_0, ax_1, ax_2, ax_3] = plt.subplots(1, 4, figsize=(20, 7))
probplot(train_no_invalid['X'], plot=ax_0)
probplot(test_no_invalid['X'], plot=ax_1)
probplot(train_no_invalid['Y'], plot=ax_2)
probplot(test_no_invalid['Y'], plot=ax_3)
plt.show()

# Kolmogorov-Smirnov
# The null-hypothesis for the KT test is that the distributions are the same
# Thus, the lower your p value -> conclude the distributions are different
display( kstest(train_no_invalid['X'], 'norm') )  # p=0
display( kstest(test_no_invalid['Y'], 'norm') )  # p=0
display( kstest(train_no_invalid['X'], 'norm') )  # p=0
display( kstest(test_no_invalid['Y'], 'norm') )  # p=0


# In[13]:


# Explore 'Dates' feature

def hist_by_groupby_valuecounts(dataset_df, col_name_to_groupby):
    col_valuecounts = dataset_df.groupby(by=col_name_to_groupby).size()
    plt.bar(col_valuecounts.index, col_valuecounts); plt.show()


# In[14]:


# Intermediate arrays: exploring 'Dates' feature

train_eda_dates = raw_train_df.copy()
train_eda_dates['Dates'] = pd.to_datetime(train_eda_dates['Dates'])

test_eda_dates = raw_test_df.copy()
test_eda_dates['Dates'] = pd.to_datetime(test_eda_dates['Dates'])


# In[15]:


# 1. Hours

# Conclusion: 'Hour' looks like a good feature

# Training set
train_eda_dates['Hour'] = train_eda_dates['Dates'].dt.hour

hist_by_groupby_valuecounts(train_eda_dates, 'Hour')

# Test set
test_eda_dates['Hour'] = test_eda_dates['Dates'].dt.hour

hist_by_groupby_valuecounts(test_eda_dates, 'Hour')


# In[16]:


# 2. Minutes

# Note: Minutes are rigged because of lot of 0, 15, 30, 45, 60 values in reports -
# this might be because to human factor.

# Conclusion: don't use 'Minutes' as a feature

# Training set
train_eda_dates['Minutes'] = train_eda_dates['Dates'].dt.minute

hist_by_groupby_valuecounts(train_eda_dates, 'Minutes')

# Test set
test_eda_dates['Minutes'] = test_eda_dates['Dates'].dt.minute

hist_by_groupby_valuecounts(test_eda_dates, 'Minutes')


# In[17]:


# 3. Month

# Conclusion: try out Month as a feature

# Training set
train_eda_dates['Month'] = train_eda_dates['Dates'].dt.month

hist_by_groupby_valuecounts(train_eda_dates, 'Month')

# Test set
test_eda_dates['Month'] = test_eda_dates['Dates'].dt.month

hist_by_groupby_valuecounts(test_eda_dates, 'Month')


# In[18]:


# 4. Year

# Conclusion: try out Year as a feature in baseline model

train_eda_dates['Dates'].dt.year.hist(bins=13); plt.show()

test_eda_dates['Dates'].dt.year.hist(bins=13); plt.show()


# In[19]:


# 5. Day of the week

# Note: feature 'DayOfWeek' already exists in both training and test sets as a str.
# Try to replace it with a numerical value.

# Conclusion: 
# Try out IsWednesday/IsFriday/IsSaturday features (highest crime rate)
# Tru out IsSunday feature (lowest crime rate) 

train_eda_dates['Dates'].dt.dayofweek.hist(bins=7); plt.show()

test_eda_dates['Dates'].dt.dayofweek.hist(bins=7); plt.show()


# In[20]:


# 6. Week of year

# Note: test_eda_dates : last week of year has a spike while train_eda_dates does not

# Conclusion: might be a feature, try it out

display(
    len(
        pd.unique(train_eda_dates['Dates'].dt.weekofyear)
    )
)

train_eda_dates['Dates'].dt.weekofyear.hist(bins=26); plt.show()

test_eda_dates['Dates'].dt.weekofyear.hist(bins=26); plt.show()


# In[39]:


# 7. quantile cuts: which hours represent biggest crime rate - weird results for now, can't understand it

train_hour_valuecnts = train_eda_dates['Hour'].value_counts()

display(train_hour_valuecnts.sort_values())

# plt.bar(
#     train_hour_valuecnts.index, train_hour_valuecnts
# )
# plt.show()

plt.bar(
    train_hour_valuecnts, train_hour_valuecnts.index
)
plt.show()

display(
    pd.qcut(train_hour_valuecnts, 3, retbins=True)[1]
)


# In[21]:


# Cleanup for train/test _eda_dates dataframes

# del train_eda_dates
# del test_eda_dates


# In[22]:


# Explore 'Category' feature from the raw_train_df

# Note the skewness of the distribution of different crime types

raw_train_df.groupby(by='Category').size().sort_values()


# In[23]:


# Explore 'Descript' feature from the raw_train_df

# This feature might be helpful when creating features 
# related to "how well police behaved in a X district" => how bad district (?)

# Conclusion: because this feature is not in the test set, drop it when creating baseline model.

display(
    raw_train_df[ ['Descript', 'Category'] ].sample(20)
)


# In[24]:


# Explore 'PdDistrict' feature

# Conclusion: 'PdDistrict' should be a good feature - definitely use it in modelling

# Training and Test sets
fig, [ax_0, ax_1] = plt.subplots(1, 2, figsize=(20, 5))
raw_train_df['PdDistrict'].sort_values().hist(bins=10, ax=ax_0)
plt.xticks(rotation=90)
raw_test_df['PdDistrict'].sort_values().hist(bins=10, ax=ax_1)
plt.xticks(rotation=90)
plt.show()


# In[25]:


# Explore 'Resolution' feature

# It is weird that there are so many rows with 'Resolution'='NONE' (526790 rows)

# We might drop this feature or use it to identify some district as good/bad,
# i.e. lots of arrests & cited - good one; lots of arrests & booked - bad one
# OR
# 'NONE' might (or not) event represent false call

for district_name, district_entries in raw_train_df.groupby(by='PdDistrict'):
    resolutions_in_district = district_entries.groupby(by='Resolution').size()
#     display(
#         resolutions_in_district.sum() -
#         resolutions_in_district[ resolutions_in_district > 1000 ].sum()  # losing 3k-5k elements
#     )
    resolutions_in_district = resolutions_in_district[ resolutions_in_district > 1000 ]
    plt.pie(
        resolutions_in_district, labels=resolutions_in_district.index,
        autopct='%1.1f%%', startangle=90
    )
    plt.title(district_name)
    plt.show()
    
display(
    raw_train_df.groupby(by='Resolution').size().sort_values()
)

display(
    raw_train_df[ ['Resolution', 'Category', 'PdDistrict'] ].sample(20)
)


# In[26]:


# Explore 'Address' feature

# Note: almost alll top Addresses by crime rate have 'Block' in their description.
# Note: intersections also identify top addresses by crime rate

# Conclusion: transform 'Address' into 'IsBlock' and 'IsIntersection'

# Note: 800 Block of BRYANT ST has top crime rate in train and test set

# Conclusion: add 'Is BryantSt800Blk' feature

display(
    len(raw_train_df['Address'].unique()), len(raw_test_df['Address'].unique())
)

display(
    raw_train_df['Address'].value_counts().head(3),
    raw_train_df['Address'].value_counts().tail(3)
)
display(
    raw_test_df['Address'].value_counts().head(3),
    raw_test_df['Address'].value_counts().tail(3),
)


# In[27]:


# Intermediate DF for fixing features

fixd_train_df = raw_train_df.copy()

fixd_test_df = raw_test_df.copy()

display(
    fixd_train_df.size, fixd_test_df.size
)


# In[28]:


# Fix rows with unusual Longtitude and Latitude == fix ['X', 'Y'] features values

# For rows that have Y=90.0 but where similiar ADDRESSES have valid coordinates:
# Replace coordinates with the same coordinates

train_invalid_rows = raw_train_df[ raw_train_df['Y'] == 90.0 ]
test_invalid_rows = raw_test_df[ raw_test_df['Y'] == 90.0 ]

def _ugly_fix_invalid_coords_inplace(invalid_rows_df, tofix_df):
    """note: used global variable concat_no_invalid_rows"""
    concat_no_invalid_rows = raw_concat_traintest_df[ raw_concat_traintest_df['Y'] != 90.0 ]
    for row_idx, row in invalid_rows_df.iterrows():
        addr_occurences_in_concat = concat_no_invalid_rows[
            concat_no_invalid_rows['Address'] == row['Address']
        ]
        if addr_occurences_in_concat.shape[0]:
            # Fix longtitude
            tofix_df.iloc[row_idx, tofix_df.columns.get_loc('X')] = addr_occurences_in_concat['X'].iloc[0]
            # Fix latitude
            tofix_df.iloc[row_idx, tofix_df.columns.get_loc('Y')] = addr_occurences_in_concat['Y'].iloc[0]

_ugly_fix_invalid_coords_inplace(train_invalid_rows, fixd_train_df)  # 67 invalid rows -> 61 invalid rows
_ugly_fix_invalid_coords_inplace(test_invalid_rows, fixd_test_df)  # 76 invalid rows -> 65 invalid rows

# Otherwise: replace with most common value.

def _ugly_fix_invalid_coords_inplace_2(tofix_df):
    tofix_df.loc[ tofix_df['Y'] == 90.0, 'X' ] = tofix_df['X'].mode()[0]  # note: because we use 'Y'=90, do X first
    tofix_df.loc[ tofix_df['Y'] == 90.0, 'Y' ] = tofix_df['Y'].mode()[0]

_ugly_fix_invalid_coords_inplace_2(fixd_train_df)
_ugly_fix_invalid_coords_inplace_2(fixd_test_df)


# In[29]:


overview_invalid_long_lat(fixd_train_df)  # should be 0

overview_invalid_long_lat(fixd_test_df)  # should be 0


# In[64]:


# Intermediate array for performing feature engineering / features dropping

feateng_train_df = fixd_train_df.copy()

feateng_test_df = fixd_test_df.copy()

# Cleanup old intermediate DFs
# del fixd_train_df
# del fixd_test_df


# In[65]:


def date_col_to_datetime_inplace(dataset_df, date_col_name='Dates'):
    dataset_df[date_col_name] = pd.to_datetime(dataset_df[date_col_name])

    
def add_date_features_inplace(dataset_df, date_col_name='Dates'):
    # Time
    dataset_df['Hour'] = dataset_df[date_col_name].dt.hour
    dataset_df['IsQuietTime'] = 0
    dataset_df.loc[ (dataset_df['Hour'] >= 1) & (dataset_df['Hour'] <= 6), 'IsQuietTime' ] = 1
    dataset_df['IsDangerousTime'] = 0
    dataset_df.loc[ (dataset_df['Hour'] >= 15) & (dataset_df['Hour'] <= 19), 'IsDangerousTime' ] = 1
    dataset_df['IsMidnight'] = 0
    dataset_df.loc[ (dataset_df['Hour'] == 0), 'IsMidnight' ] = 1
    dataset_df['IsLunchTime'] = 0
    dataset_df.loc[ (dataset_df['Hour'] == 12), 'IsLunchTime' ] = 1
    # Date: general
    dataset_df['Day'] = dataset_df[date_col_name].dt.day
    dataset_df['Month'] = dataset_df[date_col_name].dt.month
    dataset_df['Year'] = dataset_df[date_col_name].dt.year
    # Date: other
    dataset_df['DayOfWeek'] = dataset_df[date_col_name].dt.dayofweek  # Overwrite raw 'DayOfWeek' feature
    dataset_df['WeekOfYear'] = dataset_df[date_col_name].dt.weekofyear
    dataset_df['IsWeekend'] = 0
    dataset_df.loc[ dataset_df['DayOfWeek'] >= 5, 'IsWeekend' ] = 1
    # Certain "unusual risk" days    
    dataset_df['IsMiddleOfWeek'] = 0
    dataset_df.loc[ dataset_df['DayOfWeek'] == 2, 'IsMiddleOfWeek' ] = 1
    dataset_df['IsFriday'] = 0  # highest rate of crime
    dataset_df.loc[ dataset_df['DayOfWeek'] == 4, 'IsFriday' ] = 1
    dataset_df['IsSunday'] = 0  # if crime happened even on Sundays - very gangerous one  # lower rate of crime
    dataset_df.loc[ dataset_df['DayOfWeek'] == 6, 'IsSunday' ] = 1


# In[66]:


date_col_to_datetime_inplace(feateng_train_df)
# display(feateng_train_df.dtypes)  # Dates: datetime64[ns]

date_col_to_datetime_inplace(feateng_test_df)
# display(feateng_test_df.dtypes)  # Dates: datetime64[ns]


# In[67]:


add_date_features_inplace(feateng_train_df)

add_date_features_inplace(feateng_test_df)


# In[76]:


# Explore newly created features

display(
    "IsQuietTime", feateng_train_df['IsQuietTime'].value_counts()
)

display(
    "IsDangerousTime", feateng_train_df['IsDangerousTime'].value_counts()  # might be a bad one
)

display(
    "IsMidnight", feateng_train_df['IsMidnight'].value_counts()
)

display(
    "IsLunchTime", feateng_train_df['IsLunchTime'].value_counts()
)

display(
    "IsWeekend", feateng_train_df['IsWeekend'].value_counts()
)

display(
    "IsMiddleOfWeek", feateng_train_df['IsMiddleOfWeek'].value_counts()
)

display(
    "IsFriday", feateng_train_df['IsFriday'].value_counts()
)

display(
    "IsSunday", feateng_train_df['IsSunday'].value_counts()
)


# In[ ]:


def add_address_features_inplace(dataset_df, addr_col_name='Address'):
    dataset_df['IsBlock'] = 0
    
    dataset_df['IsIntersection'] = 0
    
    dataset_df['IsBryantSt800Blk'] = 0
    
display(
    
)


# In[ ]:





# In[ ]:





# In[ ]:


# Intermediate DFs : dropping features

dropfeat_train_df = feateng_train_df.copy()
dropfeat_test_df = feateng_test_df.copy()

# Cleanup
# del feateng_train_df
# del feateng_test_df


# In[ ]:


dropfeat_train_df = dropfeat_train_df.drop([''], axis=1)

dropfeat_test_df


# In[ ]:


# Intermediate DFs for baseline model

baseline_train_df = dropfeat_train_df.copy()
baseline_test_df = dropfeat_test_df.copy()

# Cleanup old intermediate DFs
# del feateng_train_df
# del feateng_test_df


# In[ ]:





# In[ ]:





# In[ ]:





# In[30]:


# todo: review as histograms columns which you didn't before


# todo: qcut for hour to check if IsRush/IsQuiet time are worth it

# timefeatures_train_df['Hour'].hist(bins=24); plt.show()
# timefeatures_test_df['Hour'].hist(bins=24); plt.show()

# todo: ismorning
# todo: isnight

# timefeatures_train_df['Day'].hist(bins=7); plt.show()
# timefeatures_test_df['Day'].hist(bins=7); plt.show()

# timefeatures_train_df['Month'].hist(bins=12); plt.show()
# timefeatures_test_df['Month'].hist(bins=12); plt.show()

# todo: isWinter

# # Note: 2014 - small amount of reported crimes
# timefeatures_train_df['Year'].hist(bins=13); plt.show()
# timefeatures_test_df['Year'].hist(bins=13); plt.show()

# timefeatures_train_df['DayOfWeek'].hist(bins=7); plt.show()
# timefeatures_test_df['DayOfWeek'].hist(bins=7); plt.show()

# timefeatures_train_df['WeekOfYear'].hist(bins=26); plt.show()
# timefeatures_test_df['WeekOfYear'].hist(bins=27); plt.show()

# timefeatures_train_df['IsWeekend'].hist(bins=2); plt.show()
# timefeatures_test_df['IsWeekend'].hist(bins=2); plt.show()

# timefeatures_train_df['IsFriday'].hist(bins=2); plt.show()
# timefeatures_test_df['IsFriday'].hist(bins=2); plt.show()

# timefeatures_train_df['IsMiddleOfWeek'].hist(bins=2); plt.show()
# timefeatures_test_df['IsMiddleOfWeek'].hist(bins=2); plt.show()

# timefeatures_train_df['IsSunday'].hist(bins=2); plt.show()
# timefeatures_test_df['IsSunday'].hist(bins=2); plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[47]:


# Intermediate DFs

baseline_train_df = timefeatures_train_df.copy()
baseline_test_df = timefeatures_test_df.copy()


# In[48]:


# Encode 'Category' feature

cat_lbl_encoder = LabelEncoder()
baseline_train_df['CategoryLblEnc'] = cat_lbl_encoder.fit_transform(baseline_train_df['Category'])


# In[49]:


# Encode 'PdDistrict' feature

pddistrict_lbl_encoder = LabelEncoder()

baseline_train_df['PdDistrict'] = pddistrict_lbl_encoder.fit_transform(baseline_train_df['PdDistrict'])

baseline_test_df['PdDistrict'] = pddistrict_lbl_encoder.fit_transform(baseline_test_df['PdDistrict'])


# In[50]:


# Prepare baseline X and y from train_set
baseline_y = baseline_train_df['CategoryLblEnc']

baseline_X = baseline_train_df.drop(
    ['Dates', 'Category', 'Descript', 'Resolution', 'Address', 'X', 'Y', 'CategoryLblEnc'], axis=1
)


# In[59]:


# Train-test split

TRAIN_SIZE = 0.7

X_tr, X_val, y_tr, y_val = train_test_split(baseline_X, baseline_y, train_size=TRAIN_SIZE, random_state=42)


# In[65]:


# Create a baseline model

# xgbclf = XGBClassifier(n_estimators=100, reg_alpha=0.05, n_jobs=-1, verbosity=2)
# xgbclf.fit(X_tr, y_tr)

# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier


# In[64]:


knn_clf = KNeighborsClassifier(n_neighbors=10, n_jobs=-1)
knn_clf = knn_clf.fit(X_tr, y_tr)


# In[67]:


y_pred = knn_clf.predict(X_val)


# In[75]:


display(y_val.values)
display(y_pred)

display(accuracy_score(y_val, y_pred))


# In[20]:


# todo: work with 'Descript' 

# display(
#     timefeatures_train_df['Descript'].unique()
# )

# display(
#     timefeatures_train_df[
#         (timefeatures_train_df['Descript'] == 'WARRANT ARREST') &
#         (timefeatures_train_df['Category'] != 'WARRANTS')
#     ]
# )  # none


# In[21]:


# todo: work with 'Address'

# todo: identify RATING for each 'Address' depending on # of crimes on that street.


# In[ ]:


# todo: work with 'PdDistrict'


# In[22]:


# todo: plot different features with color='Category' on scatter plots / differnt kinds of charts

# i.e. district by category type


# In[23]:


# todo: encode 'Category' feature


# In[ ]:


# todo: https://www.youtube.com/watch?v=ttVhB8ET5M8


# In[ ]:


# todo: apply Prophet to identify SEASONAL patterns!!!


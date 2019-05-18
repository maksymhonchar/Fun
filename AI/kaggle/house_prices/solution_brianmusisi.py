#%% [markdown]
# src: http://brianmusisi.com/design/Predicting+House+Prices-2.html

#%% Load libraries
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from scipy.special import boxcox1p
from scipy.stats import norm, skew, probplot

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

#%% Load CSVs
csv_dataset_path = "/home/max/Documents/learn/learnai/house_prices/data/{0}"
train_set = pd.read_csv(csv_dataset_path.format('train.csv'))
test_set = pd.read_csv(csv_dataset_path.format('train.csv'))

#%% Preprocess the data
train_set_data, train_set_target = train_set.iloc[:, :-1], train_set.iloc[:,-1]

# Non-categorical features
continuous_columns = ['LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea',
                      'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
                      'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
                      'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch',
                      '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'YrSold']

# Preprocess numerical features
preprocessed_continuous = np.log(train_set_data[continuous_columns].fillna(0) + 1)

# Categorical features
categorical_columns = [column for column in train_set_data.columns if column not in continuous_columns]
categorical = train_set_data[categorical_columns]

# Preprocess categorical features
dummy_columns =[]
for column in categorical_columns:
    column_dummies = pd.get_dummies(categorical[column].fillna(-99))
    categorical.drop(column, axis=1, inplace=True)
    dummy_columns.append(column_dummies)
preprocesssed_categorical = pd.concat(dummy_columns, axis=1)

# Create combined numerical&categorical preprocessed features dataframe
train_features = pd.concat([preprocessed_continuous, preprocesssed_categorical], axis=1, sort=False)

print(train_features.head())

#%% Train & Evaluate base model: Lasso
kfold = KFold(n_splits=3)

#%% Try Lasso on base model
results = cross_val_score(LinearRegression(), train_features, train_set_target, cv=kfold, n_jobs=-1)
print(results.mean())  # r2 score; 0.8217204003724916
results = cross_val_score(LinearRegression(), train_features, train_set_target, cv=kfold, scoring='neg_mean_squared_error', n_jobs=-1)
print(np.sqrt(np.abs(results.mean())))  # RMSE score; 33704.49936567566

#%% Try Ridge on base model
results = cross_val_score(RidgeCV(), train_features, train_set_target, cv=kfold, n_jobs=-1)
print(results.mean())  # 0.8413127593450854
results = cross_val_score(RidgeCV(), train_features, train_set_target, cv=kfold, scoring='neg_mean_squared_error', n_jobs=-1)
print(np.sqrt(np.abs(results.mean())))  # 31832.11311018801

#%% Try 30 trees RandomForestRegressor
results = cross_val_score(RandomForestRegressor(n_estimators=30), train_features, train_set_target, cv=kfold, n_jobs=-1)
print(results.mean())  # 0.8424253430554698
results = cross_val_score(RandomForestRegressor(n_estimators=30), train_features, train_set_target, cv=kfold, scoring='neg_mean_squared_error', n_jobs=-1)
print(np.sqrt(np.abs(results.mean())))  # 31421.54821532371

#%% Try 100 trees RandomForestRegressor
results = cross_val_score(RandomForestRegressor(n_estimators=100), train_features, train_set_target, cv=kfold, n_jobs=-1)
print(results.mean())  # 0.8526151483358477
# results = cross_val_score(RandomForestRegressor(n_estimators=100), train_features, train_set_target, cv=kfold, scoring='neg_mean_squared_error', n_jobs=-1)
# print(np.sqrt(np.abs(results.mean())))  # ?

#%% Remove the noisy points
plt.scatter(train_set['GrLivArea'], train_set['SalePrice'])
plt.xlabel('GrLivArea')
plt.ylabel('SalePrice')

train_set = train_set.drop(train_set[(train_set['GrLivArea'] > 4000) & (train_set['SalePrice'] < 300000)].index)
print(train_set.shape)

#%% Examine distribution of SalePrice: target variable
mu, sigma = norm.fit(train_set['SalePrice'])
sns.distplot(train_set['SalePrice'], fit=norm)
plt.legend(['Normal distribution (mu={:.2f} and sigma={:.2f})'.format(mu, sigma)], loc='best')
plt.ylabel('Frequency')
plt.title('Histogram of SalePrice')

fig = plt.figure()
# Probability plot of sample data against the quantiles of a specified theoretical distribution (the normal distribution by default).
# probplot optionally calculates a best-fit line 
# probability plot that shows that the variable diverges from the normal distribution.
res = probplot(train_set['SalePrice'], plot=plt)

#%% Fix distribution of SalePrice
train_set['SalePrice'] = np.log1p(train_set['SalePrice'])  # log(1 + x)

mu, sigma = norm.fit(train_set['SalePrice'])
sns.distplot(train_set['SalePrice'], fit=norm)
plt.legend(['Normal distribution (mu={:.2f} and sigma={:.2f})'.format(mu, sigma)], loc='best')
plt.ylabel('Frequency')

# result: data is more normalized and it estimates a normal distribution of the data
fig = plt.figure()
res = probplot(train_set['SalePrice'], plot=plt)

#%% Feature engineering
traintest_set = pd.concat([train_set, test_set], sort=False).reset_index(drop=True)
traintest_set.drop(['Id', 'SalePrice'], axis=1, inplace=True)

#%% Examine missing data
# Which features have missing values (NaN values) and which are worst affected to enable us figure out what we should do in each case.
traintest_set_na = (traintest_set.isnull().sum() / traintest_set.shape[0]) * 100  # get the percentage of NaN values for each feature
traintest_set_na = traintest_set_na[traintest_set_na != 0].sort_values(ascending=False)  # remove the features that don't have any Nan values and sort in descending order  
print(traintest_set_na.head())

nan_data = pd.DataFrame({'Nan Ratio': traintest_set_na})
print(nan_data.head(20))

f, ax = plt.subplots(figsize=(10,8))
sns.barplot(x=nan_data.head(30).index, y=nan_data.head(30)['Nan Ratio'])
plt.xticks(rotation=90)
plt.ylabel('Percentage of Missing data in the Feature')
plt.xlabel('Features')

#%% Fix missing data
# Missing values that indicate absence of a feature - replace NaN with 'None' or 0
replace_with_none = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
                     'GarageCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType', 'MSSubClass']
for col in replace_with_none:
    traintest_set[col].fillna('None', inplace=True)
replace_with_zero = ['GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1',
                     'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea']
for col in replace_with_zero:
    traintest_set[col].fillna(0, inplace=True)

# Missing data with the mode(): values do not indicate the lack of a feature, use most common value.
replace_with_mode = ['MSZoning', 'Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType']
for col in replace_with_mode:
    traintest_set[col].fillna(traintest_set[col].mode()[0], inplace=True)

# Missing data with the median(): similar values for continuous data
traintest_set.loc[:, 'LotFrontage'] = traintest_set.groupby(
    'Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

# Other missing values
traintest_set['Functional'].fillna('Typ', inplace=True)  # Typ=Typical

# print(traintest_set['Utilities'].fillna('Null').value_counts())
traintest_set.drop('Utilities', axis=1, inplace=True)  # almost all are 'AllPub' - remove this feature

# check for any NaN values left
traintest_set_na = (traintest_set.isnull().sum() / traintest_set.shape[0])*100
traintest_set_na = traintest_set_na[traintest_set_na != 0].sort_values(ascending=False)
print(traintest_set_na.head())
nan_data = pd.DataFrame({'Nan Ratio': traintest_set_na})
print(nan_data)

#%% Transform some numerical features to categorical features
for col in ['MSSubClass', 'OverallCond', 'YrSold', 'MoSold']:
    traintest_set.loc[:, col] = traintest_set[col].astype(str)

#%% Add additional feature
traintest_set['TotalSF'] = traintest_set['TotalBsmtSF'] + traintest_set['1stFlrSF'] + traintest_set['2ndFlrSF']

#%% Deal with skewed features
# Use Box Cox transformation to reduce the skewedness of skewed features
numerical = traintest_set.dtypes[traintest_set.dtypes != 'object'].index
skewness = traintest_set[numerical].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness_df = pd.DataFrame({'Skewness': skewness})

# use scipy's boxcox1p function, in essence computing a 1+x Box-Cox transformation
# the threshold for "skewness" that necessitates transforamtion is 0.75.
boxcox_features = skewness_df[np.abs(skewness_df['Skewness'])>0.75].index
lam = 0.15
for col in boxcox_features:
    traintest_set.loc[:, col] = boxcox1p(traintest_set[col], lam)

skewness2 = traintest_set[numerical].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness_df2 = pd.DataFrame({'Skewness': skewness2})
print(skewness_df2.head(10))

#%% Analyze correlation
train_end = train_set.shape[0]
train_set = pd.concat([traintest_set.iloc[:train_end, :], train_set['SalePrice']], axis=1)
train_corr = train_set.corr()

mask = np.zeros_like(train_corr)
mask[np.triu_indices_from(mask)] = True

cmap = sns.diverging_palette(180, 30, as_cmap=True)
with sns.axes_style("white"):
    fig, ax = plt.subplots(figsize=(13,11))
    sns.heatmap(train_corr, vmax=.8, mask=mask, cmap=cmap, cbar_kws={'shrink':.5}, linewidth=.01)

#%% Create dummy variables using one-hot encoding
traintest_set_dummy = pd.get_dummies(traintest_set)

#%% Split data again into train and test for modelling
train = traintest_set_dummy.iloc[:train_end, :]
test = traintest_set_dummy.iloc[train_end:, :]

#%% Create a model & predictions
# fix remaining nan values
print(train_set.isnull().sum().sum())
train_set['SalePrice'].fillna(train_set['SalePrice'].mode()[0], inplace=True)
print(train_set.isnull().sum().sum())

y_train = train_set['SalePrice'].values

ridge_results = cross_val_score(RidgeCV(), train.values, y_train, cv=kfold, scoring='neg_mean_squared_error', n_jobs=-1)
print('RMSE for Ridge regression is {0}\n'.format(np.sqrt(np.abs(ridge_results.mean()))))

lasso_results = cross_val_score(LassoCV(), train.values, y_train, cv=kfold, scoring='neg_mean_squared_error', n_jobs=-1)
print('RMSE for Lasso regression is {0}\n'.format(np.sqrt(np.abs(lasso_results.mean()))))

rf30_results = cross_val_score(RandomForestRegressor(n_estimators=30), train.values, y_train, cv=kfold, scoring='neg_mean_squared_error', n_jobs=-1)
print('RMSE for Ridge regression is {0}\n'.format(np.sqrt(np.abs(rf30_results.mean()))))

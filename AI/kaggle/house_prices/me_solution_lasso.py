#%% [markdown]
# src used
# https://www.kaggle.com/neviadomski/how-to-get-to-top-25-with-simple-model-sklearn
# https://www.kaggle.com/goldens/regression-top-20-with-a-very-simple-model-lasso

# %% Load libraries
import pandas as pd
import numpy as np

from sklearn.linear_model import Lasso, LassoCV, RidgeCV, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import RobustScaler

import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

# %% Load datasets
csv_dataset_path = "/home/max/Documents/learn/learnai/house_prices/data/{0}"
train_set = pd.read_csv(csv_dataset_path.format("train.csv"))
test_set = pd.read_csv(csv_dataset_path.format("test.csv"))

# %% Describe loaded datasets & define helping Dataframes
print(train_set.shape)  # (1460, 81)
print(test_set.shape)  # (1459, 80)

print(train_set.head())

traintest_set = pd.concat([train_set, test_set], sort=False)

traintest_object_dtype_columns = traintest_set.select_dtypes(include='object')
traintest_numerical_dtype_columns = traintest_set.select_dtypes(include=['float', 'int'])

print('traintest_set dtypes=object:\n{0}'.format(traintest_object_dtype_columns.columns))
print('traintest_set dtypes=float,int:\n{0}'.format(traintest_numerical_dtype_columns.columns))

# %% Check for missing data
missing_data_df = pd.concat(
    [train_set.isnull().sum(), test_set.isnull().sum()],
    axis=1,
    keys=['Train', 'Test'],
    sort=False
)
missing_data_df = missing_data_df[missing_data_df.sum(axis=1) > 0]
print(missing_data_df)

print(traintest_set.select_dtypes(include='object').isnull().sum()[
      traintest_set.select_dtypes(include='object').isnull().sum() > 0])
print(traintest_set.select_dtypes(include=['float', 'int']).isnull().sum()[
      traintest_set.select_dtypes(include=['float', 'int']).isnull().sum() > 0])

#%% Fix missing values
def fix_missing_by_replacing_all_data():
    # Fix dtype='object' types of columns
    for col in ('Alley', 'Utilities', 'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
                'BsmtFinType1', 'BsmtFinType2', 'Electrical', 'FireplaceQu', 'GarageType',
                'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature'):
        train_set[col] = train_set[col].fillna('None')
        test_set[col] = test_set[col].fillna('None')
    for col in ('MSZoning', 'Exterior1st', 'Exterior2nd', 'KitchenQual', 'SaleType', 'Functional'):
        train_set[col] = train_set[col].fillna(train_set[col].mode()[0])
        test_set[col] = test_set[col].fillna(test_set[col].mode()[0])

    # Fix dtype=['float', 'int'] types of columns
    for col in ('MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
                'BsmtFullBath', 'BsmtHalfBath', 'GarageYrBlt', 'GarageCars', 'GarageArea'):
        train_set[col] = train_set[col].fillna(0)
        test_set[col] = test_set[col].fillna(0)
    train_set['LotFrontage'] = train_set['LotFrontage'].fillna(train_set['LotFrontage'].mean())
    test_set['LotFrontage'] = test_set['LotFrontage'].fillna(test_set['LotFrontage'].mean())


def fix_missing_by_dropping_data():
    # remove columns which contain a lot (~a half) of NaN values.
    features_to_drop = ['Utilities', 'RoofMatl', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 
                        'BsmtUnfSF', 'Heating', 'LowQualFinSF', 'BsmtFullBath', 'BsmtHalfBath',
                        'Functional', 'GarageYrBlt', 'GarageArea', 'GarageCond', 'WoodDeckSF',
                        'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
                        'PoolQC', 'Fence', 'MiscFeature', 'MiscVal']
    train_set.drop(features_to_drop, axis=1, inplace=True)
    test_set.drop(features_to_drop, axis=1, inplace=True)

fix_missing_by_replacing_all_data()
# fix_missing_by_dropping_data()

print(train_set.isnull().sum().sum())  # 0
print(test_set.isnull().sum().sum())  # 0

#%% Convert several features str
column_names_to_str = ['MSSubClass', 'OverallCond', 'OverallQual', 'GarageCars']
for col in column_names_to_str:
    train_set[col] = train_set[col].astype(str)
    test_set[col] = test_set[col].astype(str)

#%% Add additional features
train_set['TotalSF'] = train_set['TotalBsmtSF'] + train_set['1stFlrSF'] + train_set['2ndFlrSF']
train_set.drop(['TotalBsmtSF', '1stFlrSF', '2ndFlrSF'], axis=1, inplace=True)

test_set['TotalSF'] = test_set['TotalBsmtSF'] + test_set['1stFlrSF'] + test_set['2ndFlrSF']
test_set.drop(['TotalBsmtSF', '1stFlrSF', '2ndFlrSF'], axis=1, inplace=True)

#%% Research correlation to SalePrice

# As a rule of thumb, for absolute value of r:
# 0.00-0.19: very weak
# 0.20-0.39: weak
# 0.40-0.59: moderate 
# 0.60-0.79: strong
# 0.80-1.00: very strong.

corr_matrix = train_set.corr()
lower_corr_matrix = corr_matrix.where(np.tril(np.ones(corr_matrix.shape)).astype(np.bool))
saleprice_corr_series = lower_corr_matrix.loc['SalePrice', :][1:]  # skip 'Id' column
saleprice_low_corr_series = saleprice_corr_series[saleprice_corr_series < 0.20]
print('Columns with low correlation to SalePrice are: {0}'.format(saleprice_low_corr_series))

plt.figure(figsize=[50, 30])
# sns.heatmap(corr_matrix, annot=True)
sns.heatmap(lower_corr_matrix, annot=True)

#%% Remove columns which are low correlated to 'SalePrice'
def remove_saleprice_low_correlated_cols():
    if not saleprice_low_corr_series.empty:
        for idx, value in saleprice_low_corr_series.items():
            train_set.drop(idx, axis=1, inplace=True)
            test_set.drop(idx, axis=1, inplace=True)

remove_saleprice_low_correlated_cols()

print(train_set.shape, test_set.shape)

#%% Check skewness
# Wiki: skewness is a measure of the asymmetry of the probability distribution of a real-valued random variable about its mean. 
train_labels = train_set.pop('SalePrice')

print('Univariate distribution of SalePrice BEFORE np.log')
ax = sns.distplot(train_labels)

#%% Fix skewness
train_labels = np.log(train_labels)

print('Univariate distribution of SalePrice AFTER np.log')
ax_2 = sns.distplot(train_labels)

#%% Std-ize numeric data: train set
# Manually select values which I want to std-ize
numeric_cols_names = ['LotFrontage', 'LotArea', 'BsmtUnfSF', 'GrLivArea', 'TotalSF']

train_numeric_cols = train_set.loc[:, numeric_cols_names]
train_numeric_cols_stdized = \
    (train_numeric_cols - train_numeric_cols.mean()) / train_numeric_cols.std()

ax = sns.pairplot(train_numeric_cols_stdized)

#%% Std-ize numeric data: test set
test_numeric_cols = test_set.loc[:, numeric_cols_names]
test_numeric_cols_stdized = \
    (test_numeric_cols - test_numeric_cols.mean()) / test_numeric_cols.std()

ax = sns.pairplot(test_numeric_cols_stdized)

#%% Convert categorical data to dummies
traintest_set = pd.concat([train_set, test_set], sort=False)
traintest_set = pd.get_dummies(traintest_set)

len_train = train_set.shape[0]

#%% Prepare the model
train = traintest_set[:len_train]
test = traintest_set[len_train:]

train = train.drop('Id', axis=1)
test = test.drop('Id', axis=1)

x = train
y = train_labels

robust_sc = RobustScaler()
x = robust_sc.fit_transform(x)
test = robust_sc.transform(test)

#%% Build the model
model = Lasso(alpha=0.001, random_state=1)
model.fit(x, y)

#%% Kaggle submission
pred = model.predict(test)
preds = np.exp(pred)

test_set_2 = pd.read_csv(csv_dataset_path.format("test.csv"))

output = pd.DataFrame( {'Id': test_set_2.Id, 'SalePrice': preds} )
output.to_csv(
    '/home/max/Documents/learn/learnai/house_prices/submission.csv', index=False)

print(output.head())

#%% Try out different models
import warnings
warnings.filterwarnings('ignore')

x_train_raw = train
y_train_raw = np.exp(train_labels)

ridge_results = cross_val_score(LassoCV(), x_train_raw.values, y_train_raw.values,
                                cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
print('RMSE for Ridge regression is {0}\n'.format(
    np.sqrt(np.abs(ridge_results.mean()))
))
print(ridge_results.mean())


# lasso_results = cross_val_score(LassoCV(), train.values, y_train, cv=kfold, scoring= 'neg_mean_squared_error')
# rf_30_results = cross_val_score(RandomForestRegressor(n_estimators=30), train.values, y_train, cv=kfold, scoring= 'neg_mean_squared_error')
# rf_100_results = cross_val_score(RandomForestRegressor(n_estimators=100), train.values, y_train, cv=kfold, scoring= 'neg_mean_squared_error')


#%% Calculate scores: r2 and RMSE
# _dummy_y = np.exp(y)  # recover train_labels from log state
# _dummy_y = _dummy_y[:-1]  # to compare with 'preds' with .shape (1459,)
# print('Scores are: r2: {0}; RMSE: {1}; mean_squared_error={2}'.format(
#     r2_score(_dummy_y, preds),
#     np.sqrt(mean_squared_error(_dummy_y, preds)),
#     mean_squared_error(_dummy_y, preds)
# ))

results = cross_val_score(LinearRegression(), x_raw, y_raw, cv=kfold)
# print(results.mean())  # r2 = 0.8022787705561434  (also called the coefficient of determination)

# results = cross_val_score(LinearRegression(), x_raw, y_raw,
#                           cv=kfold, scoring='neg_mean_squared_error')
# print(np.sqrt(np.abs(results.mean())))  # 36380.34069016587
# # RMSE score - root mean square error
# # RMSE ~= average difference between the predicted values and the actual values.

# ridge_results = cross_val_score(RidgeCV(), x_raw, y_raw,
#                                 cv=kfold, scoring='neg_mean_squared_error')
# print('RMSE for Ridge regression is %.3f\n' % np.sqrt(np.abs(ridge_results.mean())))  # 31924.019
# print(ridge_results.mean())

# lasso_results = cross_val_score(LassoCV(), x_raw, y_raw,
#                                 cv=kfold, scoring='neg_mean_squared_error')
# print('RMSE for Lasso regression is %.3f\n' % np.sqrt(np.abs(lasso_results.mean())))  # 43199.478
# print(lasso_results.mean())

# rf_30_results = cross_val_score(RandomForestRegressor(n_estimators=30), x_raw, y_raw,
#                                 cv=kfold, scoring='neg_mean_squared_error')
# print('RMSE for Random Forest regression with 30 trees is %.3f\n' % np.sqrt(np.abs(rf_30_results.mean())))  # 31008.951
# print(rf_30_results.mean())

# rf_100_results = cross_val_score(RandomForestRegressor(n_estimators=100), x_raw, y_raw,
#                                  cv=kfold, scoring='neg_mean_squared_error')
# print('RMSE for Random Forest regression with 100 trees  is %.3f' % np.sqrt(np.abs(rf_100_results.mean())))  # 31237.195
# print(rf_100_results.mean())

#%% [markdown]
# results: Top 60% 2,736th of 4593
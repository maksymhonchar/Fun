# %% [markdown]
# src: https://www.kaggle.com/goldens/regression-top-20-with-a-very-simple-model-lasso

# %% Import libraries
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.linear_model import Lasso
from sklearn.preprocessing import RobustScaler

from scipy.stats import skew
from scipy.special import boxcox1p

import matplotlib.pyplot as plt

# %% Load train and test datasets
csv_dataset_path = "/home/max/Documents/learn/learnai/house_prices/data/{0}"
train_set = pd.read_csv(csv_dataset_path.format("train.csv"))
test_set = pd.read_csv(csv_dataset_path.format("test.csv"))
traintest_set = pd.concat([train_set, test_set], sort=False)

# %% Describe loaded datasets
print(train_set.shape)  # (1460, 81)
print(test_set.shape)  # (1459, 80)
print(traintest_set.shape)  # (2919, 81)

print('traintest_set dtypes=object:\n{0}'.format(traintest_set.select_dtypes(include='object').columns))
print('traintest_set dtypes=float,int:\n{0}'.format(traintest_set.select_dtypes(include=['float', 'int']).columns))

# %% Fix missing values
print('Before fixing missing values, traintest_set:')
print(traintest_set.select_dtypes(include='object').isnull().sum()[traintest_set.select_dtypes(include='object').isnull().sum() > 0])
print(traintest_set.select_dtypes(include=['float', 'int']).isnull().sum()[traintest_set.select_dtypes(include=['float', 'int']).isnull().sum() > 0])

for col in ('Alley', 'Utilities', 'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Electrical', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature'):
    train_set[col] = train_set[col].fillna('None')
    test_set[col] = test_set.fillna('None')
for col in ('MSZoning', 'Exterior1st', 'Exterior2nd', 'KitchenQual', 'SaleType', 'Functional'):
    train_set[col] = train_set[col].fillna(train_set[col].mode()[0])
    test_set[col] = test_set[col].fillna(test_set[col].mode()[0])

for col in ('MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'GarageYrBlt', 'GarageCars', 'GarageArea'):
    train_set[col] = train_set[col].fillna(0)
    test_set[col] = test_set[col].fillna(0)
train_set['LotFrontage'] = train_set['LotFrontage'].fillna(train_set['LotFrontage'].mean())
test_set['LotFrontage'] = test_set['LotFrontage'].fillna(test_set['LotFrontage'].mean())

print('After fixing missing values, train_set:')
print(train_set.select_dtypes(include='object').isnull().sum()[train_set.select_dtypes(include='object').isnull().sum() > 0])  # Series([], dtype: int64)
print(train_set.select_dtypes(include=['float', 'int']).isnull().sum()[train_set.select_dtypes(include=['float', 'int']).isnull().sum() > 0])  # # Series([], dtype: int64)
print(train_set.isnull().sum().sum())  # 0
print('After fixing missing values, test_set:')
print(test_set.select_dtypes(include='object').isnull().sum()[test_set.select_dtypes(include='object').isnull().sum() > 0])  # # Series([], dtype: int64)
print(test_set.select_dtypes(include=['float', 'int']).isnull().sum()[test_set.select_dtypes(include=['float', 'int']).isnull().sum() > 0])  # # Series([], dtype: int64)
print(test_set.isnull().sum().sum())  # 0

# %% Remove features high correlated and outliers
# plt.figure(figsize=[50, 35])
# sns.heatmap(train_set.corr(), annot=True)

# corr_matrix = train_set.corr().abs()
# upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
# to_drop = [col for col in upper.columns if any(upper[col] > 0.65)]
# print(to_drop)

print('Before removing correlated cols:', train_set.shape, test_set.shape)

train_set = train_set.drop(['GarageArea', '1stFlrSF', 'TotRmsAbvGrd', '2ndFlrSF'], axis=1)  # (1460, 81) (1459, 80)
test_set = test_set.drop(['GarageArea', '1stFlrSF', 'TotRmsAbvGrd', '2ndFlrSF'], axis=1)  # (1460, 77) (1459, 76)

print('After removing correlated cols:', train_set.shape, test_set.shape)

# removing outliers recomended by author
train_set = train_set[train_set['GrLivArea'] < 4000]

# update traintest_set after all the deletions
print('traintest_set shape before:', traintest_set.shape)  # (2919, 81)
traintest_set = pd.concat([train_set, test_set], sort=False)
print('traintest_set shape after:', traintest_set.shape)  # (2915, 77)

# %% Transform data in the dataset
# Transform numerical to categorical
traintest_set['MSSubClass'] = traintest_set['MSSubClass'].astype(str)

# Skew
skew = traintest_set.select_dtypes(include=['int', 'float']).apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skew_df = pd.DataFrame({'Skew': skew})
skewed_df = skew_df[(skew_df['Skew'] > 0.5) | (skew_df['Skew'] < -0.5)]

print(skewed_df.index)

train_len = train_set.shape[0]
train_set = traintest_set[:train_len]
test_set = traintest_set[train_len:]

lam = 0.1
for col in ('MiscVal', 'PoolArea', 'LotArea', 'LowQualFinSF', '3SsnPorch', 'KitchenAbvGr', 'BsmtFinSF2', 'EnclosedPorch', 'ScreenPorch', 'BsmtHalfBath', 'MasVnrArea', 'OpenPorchSF', 'WoodDeckSF', 'LotFrontage', 'GrLivArea', 'BsmtFinSF1', 'BsmtUnfSF', 'Fireplaces', 'HalfBath', 'TotalBsmtSF', 'BsmtFullBath', 'OverallCond', 'YearBuilt', 'GarageYrBlt'):
    train_set[col] = boxcox1p(train_set[col], lam)
    test_set[col] = boxcox1p(test_set[col], lam)

train_set['SalePrice'] = np.log(train_set['SalePrice'])

traintest_set = pd.concat([train_set, test_set], sort=False)
traintest_set = pd.get_dummies(traintest_set)

#%% Prepare train and test sets for the model
train_set = traintest_set[:train_len]
test_set = traintest_set[train_len:]

train_set = train_set.drop('Id', axis=1)
test_set = test_set.drop('Id', axis=1)

X = train_set.drop('SalePrice', axis=1)
y = train_set['SalePrice']

test_set = test_set.drop('SalePrice', axis=1)

sc = RobustScaler()
X = sc.fit_transform(X)
test_set = sc.transform(test_set)

#%% Build the model
model = Lasso(alpha=.001, random_state=1)
model.fit(X, y)

#%% Kaggle submission
pred = model.predict(test_set)
preds = np.exp(pred)

print(model.score(X, y))

output=pd.DataFrame({'Id':test2.Id, 'SalePrice':preds})
output.to_csv('submission.csv', index=False)

output.head()

# %% [markdown]
# todos: Creative feature engineering; random forest; gradient boosting; xgboost
# todo: encode the categorical variable: try LabelEncoder or OneHotEncoder

# %% [markdown]
# visualization examples:

# train_set.hist(bins=100, figsize=(50, 30))
# plt.show()
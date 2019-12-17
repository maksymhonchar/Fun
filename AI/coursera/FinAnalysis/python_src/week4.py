#%% Load libraries
from pandas.plotting import scatter_matrix
import scipy.stats as stats
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.style
import matplotlib.pyplot as plt
%matplotlib inline
mpl.style.use('classic')


#%% Import dataset

#%% Association between random variables
housing = pd.DataFrame.from_csv('data/housing.csv')

display(housing.shape)

display(housing.columns.values)
# columns: array(['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV'], dtype=object)
# LSTAT - percentage of the population classified as low status
# INDUS - proportion of non-retail business across the town
# NOX - NO2 concentration
# RM - average number of rooms per dwelling
# MEDV - median value of the owner-occupied homes, in $1000s

display(housing.describe())
display(housing.head())

# Check out covariance
# Strongest association - ? todo to understand that: coefficient of correlation
display(housing.cov())

# Check out correlation
display(housing.corr())

#%% Visualize the association between two variables
# Scatter matrix is a matrix of scatter plots for each pair of random variabless.
# The histograms in diagonal positions are those of variables.
housing_scatter_matrix = pd.plotting.scatter_matrix(housing, figsize=(10, 10))
plt.show()

#%% Observe the association more closely
housing.plot(
    kind='scatter',
    x='LSTAT', y='MEDV',
    figsize=(5, 5)
)
plt.show()

housing.plot(
    kind='scatter',
    x='RM', y='MEDV',
    figsize=(5, 5)
)
plt.show()

#%% Simple linear regression model

# y_i = betha_0 + betha_1 * x_i + eps_i

# Try to estimate relations between RM (OX) and MEDV (OY)

# Step 1. estimate values of betha_0 (intercept) and betha_1 (slope)

# Guess of b0 and b1
b0 = 0.1
b1 = 1
housing['guess_response'] = b0 + b1 * housing['RM']
# Calculate error of the guess; this shows how far is our guess response from the true response
housing['guess_error'] = housing['MEDV'] - housing['guess_response']
# Plot estimated line together with real points' values
plt.figure(figsize=(8, 8))
plt.scatter(
    x=housing['RM'], y=housing['MEDV'],
    color='g', label='Observed values'
)
plt.plot(
    housing['RM'], housing['guess_response'],
    color='red', label='guessed response (b0=0.1, b1=1)'
)
plt.legend()
plt.show()
# Calculate SSE
sse_value = (housing['guess_error']**2).sum()
display('Sum of squared error is {0}'.format(sse_value))

#%% Least square estimates
formula = 'MEDV~RM'
model = smf.ols(formula=formula, data=housing).fit()

display(model.params)

b0_ols = model.params[0]
b1_ols = model.params[1]

housing['best_response'] = b0_ols + b1_ols * housing['RM']

housing['best_error'] = housing['MEDV'] - housing['best_response']

# plot estimated line
plt.figure(figsize=(8, 8))
plt.scatter(
    x=housing['RM'], y=housing['MEDV'],
    color='g', label='Observed values'
)
plt.plot(
    housing['RM'], housing['best_response'],
    color='red', label='best response (b0=0.1, b1=1)'
)
plt.legend()
plt.show()

# Calculate SSEs
guess_sse_value = (housing['guess_error']**2).sum()
best_sse_value = (housing['best_error']**2).sum()

display('Sums of squared error are {0}'.format(
    [guess_sse_value, best_sse_value]))
# 'Sums of squared error are [170373.528047, 22061.8791962118]'  # much lower in the second case!!!

#%% Print model summary
model.summary()
# R-squared:	0.484
# F-statistic:	471.8

#%% Diagnostic of linear regression model

# Assumptions of Linear Regression Model
# linearity
# independence
# normality
# equal variance

#%% linearity
# in our case, scatter plot between MEDV and RM has linear pattern - OK

#%% Independence
# Demonstrate that observed error is independent mutually
housing['error'] = housing['MEDV'] - housing['best_response']
plt.figure(figsize=(10, 5))
plt.plot(housing.index, housing['error'], color='red')
plt.axhline(y=0, color='red')
plt.show()  # there is no obvious pattern in the plot

# Durbin Watson test for serial correlation
# Durbin-Watson:	0.684  -- bad score

#%% Normality
# use quantile-quantile plot (QQ plot)
z = (housing['error'] - housing['error'].mean()) / housing['error'].std(ddof=1)
stats.probplot(z, dist='norm', plot=plt)
plt.title('Normal q-q plot')
plt.show()

#%% Equal variance
housing.plot(
    kind='scatter',
    x='RM', y='error',
    figsize=(10, 5), color='green'
)
plt.title('Residuals vs predictor')
plt.axhline(y=0, color='red')
plt.show()


housing.plot(
    kind='scatter',
    x='LSTAT', y='error',
    figsize=(10, 5), color='green'
)
plt.title('Residuals vs predictor')
plt.axhline(y=0, color='red')
plt.show()

# We can see that the regression model (MEDV~LSTAT) violates all four assumptions. Therefore, we cannot make statistical inference using this model
# However, the model can still be applied to make a prediction. The accuracy and the consistency of your model, do not rely on these four assumptions.

#%% Multiple linear regression model
# mimic the process of building trading model of SPY, using historical data of DIFFERENT stock markets.
aord = pd.DataFrame.from_csv('data/ALLOrdinary.csv')
nikkei = pd.DataFrame.from_csv('data/Nikkei225.csv')
hsi = pd.DataFrame.from_csv('data/HSI.csv')
daxi = pd.DataFrame.from_csv('data/DAXI.csv')
cac40 = pd.DataFrame.from_csv('data/CAC40.csv')
sp500 = pd.DataFrame.from_csv('data/SP500.csv')
dji = pd.DataFrame.from_csv('data/DJI.csv')
nasdaq = pd.DataFrame.from_csv('data/nasdaq_composite.csv')
spy = pd.DataFrame.from_csv('data/SPY.csv')

display(nasdaq.head())
# ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
display(nasdaq.columns.values)

#%% use timezones

indicepanel = pd.DataFrame(index=spy.index)

indicepanel['spy'] = spy['Open'].shift(-1)-spy['Open']
indicepanel['spy_lag1'] = indicepanel['spy'].shift(1)
indicepanel['sp500'] = sp500["Open"]-sp500['Open'].shift(1)
indicepanel['nasdaq'] = nasdaq['Open']-nasdaq['Open'].shift(1)
indicepanel['dji'] = dji['Open']-dji['Open'].shift(1)

indicepanel['cac40'] = cac40['Open']-cac40['Open'].shift(1)
indicepanel['daxi'] = daxi['Open']-daxi['Open'].shift(1)

indicepanel['aord'] = aord['Close']-aord['Open']
indicepanel['hsi'] = hsi['Close']-hsi['Open']
indicepanel['nikkei'] = nikkei['Close']-nikkei['Open']
indicepanel['Price'] = spy['Open']

#%% fix NaN values
display(indicepanel.isnull().sum())

# fill NaNs with forward values

indicepanel = indicepanel.fillna(method='ffill')
indicepanel = indicepanel.dropna()

# check for NaNs again
display(indicepanel.isnull().sum())

# save cleaned data
path_save = 'indicepanel.csv'
indicepanel.to_csv(path_save)

display(indicepanel.shape)

#%% Split dataset & explore train and test sets of data
# because this is financial data (== high noise level), divide into equal parts
Train = indicepanel.iloc[-2000:-1000, :]
Test = indicepanel.iloc[-1000:, :]

print(Train.shape, Test.shape)

# sm = scatter_matrix(Train, figsize=(10, 10))
# plt.show()

#%% Check out correlation of each index between SPY
corr_array = Train.iloc[:, :-1].corr()['spy']
print(corr_array)

formula = 'spy~spy_lag1+sp500+nasdaq+dji+cac40+aord+daxi+nikkei+hsi'
lm = smf.ols(formula=formula, data=Train).fit()
lm.summary()

#%% Make prediction
Train['PredictedY'] = lm.predict(Train)
Test['PredictedY'] = lm.predict(Test)

plt.scatter(Train['spy'], Train['PredictedY'])
plt.show()

#%% Evaluate the model

# RMSE - Root Mean Squared Error, Adjusted R^2

def adjustedMetric(data, model, model_k, yname):
    data['yhat'] = model.predict(data)
    SST = ((data[yname] - data[yname].mean())**2).sum()
    SSR = ((data['yhat'] - data[yname].mean())**2).sum()
    SSE = ((data[yname] - data['yhat'])**2).sum()
    r2 = SSR/SST
    adjustR2 = 1 - (1-r2)*(data.shape[0] - 1)/(data.shape[0] - model_k - 1)
    RMSE = (SSE/(data.shape[0] - model_k - 1))**0.5
    return adjustR2, RMSE


def assessTable(test, train, model, model_k, yname):
    r2test, RMSEtest = adjustedMetric(test, model, model_k, yname)
    r2train, RMSEtrain = adjustedMetric(train, model, model_k, yname)
    assessment = pd.DataFrame(index=['R2', 'RMSE'], columns=['Train', 'Test'])
    assessment['Train'] = [r2train, RMSEtrain]
    assessment['Test'] = [r2test, RMSEtest]
    return assessment

# Get the assement table fo our model
assessTable(Test, Train, lm, 9, 'spy')
# Train	Test
# R2	0.059020	0.067248
# RMSE	1.226068	1.701291

#%% Strategy built from Regression model
Train = indicepanel.iloc[-2000:-1000, :]
Test = indicepanel.iloc[-1000:, :]

formula = 'spy~spy_lag1+sp500+nasdaq+dji+cac40+aord+daxi+nikkei+hsi'
lm = smf.ols(formula=formula, data=Train).fit()

Train['PredictedY'] = lm.predict(Train)
Test['PredictedY'] = lm.predict(Test)

# Train
Train['Order'] = [1 if sig>0 else -1 for sig in Train['PredictedY']]
Train['Profit'] = Train['spy'] * Train['Order']

Train['Wealth'] = Train['Profit'].cumsum()
print('Total profit made in Train: ', Train['Profit'].sum())

plt.figure(figsize=(10, 10))
plt.title('Performance of Strategy in Train')
plt.plot(Train['Wealth'].values, color='green', label='Signal based strategy')
plt.plot(Train['spy'].cumsum().values, color='red', label='Buy and Hold strategy')
plt.legend()
plt.show()

# Test
Test['Order'] = [1 if sig>0 else -1 for sig in Test['PredictedY']]
Test['Profit'] = Test['spy'] * Test['Order']

Test['Wealth'] = Test['Profit'].cumsum()
print('Total profit made in Test: ', Test['Profit'].sum())

plt.figure(figsize=(10, 10))
plt.title('Performance of Strategy in Test')
plt.plot(Test['Wealth'].values, color='green', label='Signal based strategy')
plt.plot(Test['spy'].cumsum().values, color='red', label='Buy and Hold strategy')
plt.legend()
plt.show()

#%% Evaluate strategy
Train['Wealth'] = Train['Wealth'] + Train.loc[Train.index[0], 'Price']
Test['Wealth'] = Test['Wealth'] + Test.loc[Test.index[0], 'Price']

# Sharpe Ratio on Train data
Train['Return'] = np.log(Train['Wealth']) - np.log(Train['Wealth'].shift(1))
dailyr = Train['Return'].dropna()

print('Daily Sharpe Ratio is ', dailyr.mean()/dailyr.std(ddof=1))
print('Yearly Sharpe Ratio is ', (252**0.5)*dailyr.mean()/dailyr.std(ddof=1))

# Sharpe Ratio in Test data
Test['Return'] = np.log(Test['Wealth']) - np.log(Test['Wealth'].shift(1))
dailyr = Test['Return'].dropna()

print('Daily Sharpe Ratio is ', dailyr.mean()/dailyr.std(ddof=1))
print('Yearly Sharpe Ratio is ', (252**0.5)*dailyr.mean()/dailyr.std(ddof=1))

# Maximum Drawdown in Train data
Train['Peak'] = Train['Wealth'].cummax()
Train['Drawdown'] = (Train['Peak'] - Train['Wealth'])/Train['Peak']
print('Maximum Drawdown in Train is ', Train['Drawdown'].max())

# Maximum Drawdown in Test data
Test['Peak'] = Test['Wealth'].cummax()
Test['Drawdown'] = (Test['Peak'] - Test['Wealth'])/Test['Peak']
print('Maximum Drawdown in Test is ', Test['Drawdown'].max())
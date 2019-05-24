#%% [markdown]
# src: https://towardsdatascience.com/an-end-to-end-project-on-time-series-analysis-and-forecasting-with-python-4835e6bf050b


#%% Load libraries
import itertools
import numpy as np
import pandas as pd

import statsmodels.api as sm

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

#%% Load Superstore sales data
sales_df = pd.read_excel('/home/max/Documents/learn/learnai/time_series_analysis/superstore_sales_analysis/Sample - Superstore.xls')

#%% Display portions of dataset
print(sales_df.shape)  # (9994, 21)
print(sales_df.columns)

#%% Prepare furniture df
furniture_df = sales_df.loc[sales_df['Category'] == 'Furniture']
print(furniture_df.shape)  # (2121, 21)
print(furniture_df['Order Date'].min())  # 2014-01-06 00:00:00
print(furniture_df['Order Date'].max())  # 2017-12-30 00:00:00

furniture_order_sales_df = furniture_df[['Order Date', 'Sales']]
print(furniture_order_sales_df.shape)  # (2121, 2)

furniture_order_sales_df = furniture_order_sales_df.sort_values('Order Date')
furniture_order_sales_df = furniture_order_sales_df.groupby(
    'Order Date')['Sales'].sum().reset_index()

furniture_order_sales_df = furniture_order_sales_df.set_index('Order Date')

# Use the average daily sales value for month.
# Use the start of each month as the timestamp.
y = furniture_order_sales_df['Sales'].resample('MS').mean()

#%% Visualize furniture sales time series data
y.plot(figsize=(20, 10))
plt.show()

# some patterns appear on the plot: The time-series has seasonality pattern, such as sales are always low at the beginning of the year and high at the end of the year.

# There is always an upward trend within any single year with a couple of low months in the mid of the year.

#%% Visualize data using time-series decomposition method.
# Decompose time series into 3 distinct components: trend, seasonality, noise
decomposition = sm.tsa.seasonal_decompose(y, model='additive')
fig = decomposition.plot()
plt.show()

# The plot shows that the sales of furniture is unstable, along with its seasonality.

#%% Time series forecasting with ARIMA
# ARIMA - Autoregressive Integrated Moving Average
# ARIMA(p, d, q)  # seasonality, trend, noise in data
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))

seasonal_pdq = [(x[0], x[1], x[2], 12)
                for x in list(itertools.product(p, d, q))]

print('Examples of parameter combinations for Seasonal ARIMA:')
print('SARIMAX: {0} x {1}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {0} x {1}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {0} x {1}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {0} x {1}'.format(pdq[2], seasonal_pdq[4]))

#%% Select parameters for our furniture's sales ARIMA Time Series Model.
# Goal: use a "grid search" to find the optimal set of parameters that yields the best performance for our model.
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(
                y,
                order=param, seasonal_order=param_seasonal,
                enforce_stationarity=False, enforce_invertibility=False
            )
            results = mod.fit()
            print('ARIMA {0}x{1} 12 - AIC: {2}'.format(
                param, param_seasonal, results.aic
            ))
        except ValueError:
            # ValueError: maxlag should be < nobs
            continue

# min AIC: 297.78 with SARIMAX(1,1,1)x(1,1,0,12) - consider this to be optimal option.

#%% Fit the ARIMA model
arima_model = sm.tsa.statespace.SARIMAX(
    y,
    order=(1, 1, 1), seasonal_order=(1, 1, 0, 12),
    enforce_stationarity=False, enforce_invertibility=False
)
arima_results = arima_model.fit()

print(arima_results.summary())

#%% ARIMA model diagnostics
# Always run model diagnostics to investigate any unusual behavior.
arima_results.plot_diagnostics(figsize=(20, 10))
plt.show()

# It is not perfect, however, our model diagnostics suggests that the model residuals are near normally distributed.

#%% Validating forecasts
# To help us understand the accuracy of our forecasts, compare predicted sales to real sales of the time series.
# Set start at 2017-01-01 to the end of the data.
arima_pred = arima_results.get_prediction(
    start=pd.to_datetime('2017-01-01'),
    dynamic=False
)
arima_pred_ci = arima_pred.conf_int()

ax = y['2014':].plot(label='observed')
arima_pred.predicted_mean.plot(
    ax=ax, alpha=0.7,
    label='One-step ahead Forecast',
    figsize=(14, 7)
)
ax.fill_between(
    arima_pred_ci.index,
    arima_pred_ci.iloc[:, 0],
    arima_pred_ci.iloc[:, 1], color='w', alpha=.2
)
ax.set_xlabel('Date')
ax.set_ylabel('Furniture Sales')
plt.legend()
plt.show()

# The line plot is showing the observed values compared to the rolling forecast predictions.
# Overall, our forecasts align with the true values very well, showing an upward trend starts from the beginning of the year and captured the seasonality toward the end of the year.

#%% Calculate r2 and mean_squared_error
y_forecasted = arima_pred.predicted_mean
y_truth = y['2017-01-01':]

# MSE - Mean Squared Error - estimator that measures the average of the squares of the errors.
# MSE is a measure of the quality of an estimator - it is always non-negative, and the smaller the MSE, the closer we are to finding the line of best fit.
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The MSE of forecasts is {0}'.format(mse))  # 22993.57527123718

# RMSE - Root Mean Squared Error - tells that our model was able to forecast the average daily furniture sales in the test set within 151.64 of the real sales.
# Our furniture daily sales range from around 400 to over 1200.
print('The RMSE of forecasts is {0}'.format(np.sqrt(mse)))  # 151.6363256981558

#%% Produce and visualize forecasts
pred_uc = results.get_forecast(steps=100)
pred_ci = pred_uc.conf_int()
ax = y.plot(label='observed', figsize=(14, 7))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(
    pred_ci.index,
    pred_ci.iloc[:, 0],
    pred_ci.iloc[:, 1], color='w', alpha=.25
)
ax.set_xlabel('Date')
ax.set_ylabel('Furniture Sales')
plt.legend()
plt.show()

# As we forecast further out into the future, it is natural for us to become less confident in our values. This is reflected by the confidence intervals generated by our model, which grow larger as we move further out into the future.

#%% Research another categoty - Office Supplies
office_df = sales_df.loc[sales_df['Category'] == 'Office Supplies']
print(office_df.shape)  # (6026, 21)

# Compare Office Supplies sales with Furniture sales
furniture_order_sales_df = furniture_df[['Order Date', 'Sales']]
print(furniture_order_sales_df.shape)  # (2121, 2)
office_order_sales_df = office_df[['Order Date', 'Sales']]
print(office_order_sales_df.shape)  # (6026, 2)

furniture_order_sales_df = furniture_order_sales_df.sort_values('Order Date')
furniture_order_sales_df = furniture_order_sales_df.groupby(
    'Order Date')['Sales'].sum().reset_index()
furniture_order_sales_df = furniture_order_sales_df.set_index('Order Date')
y_furniture = furniture_order_sales_df['Sales'].resample('MS').mean()

office_order_sales_df = office_order_sales_df.sort_values('Order Date')
office_order_sales_df = office_order_sales_df.groupby(
    'Order Date')['Sales'].sum().reset_index()
office_order_sales_df = office_order_sales_df.set_index('Order Date')
y_office = office_order_sales_df['Sales'].resample('MS').mean()

furniture = pd.DataFrame(
    {'Order Date': y_furniture.index, 'Sales': y_furniture.values}
)
office = pd.DataFrame(
    {'Order Date': y_office.index, 'Sales': y_office.values}
)

furniture_office_sales = furniture.merge(office, how='inner', on='Order Date')
furniture_office_sales.rename(
    columns={'Sales_x': 'furniture_sales', 'Sales_y': 'office_sales'}, inplace=True
)
print(furniture_office_sales.head())

#%% Plot office & furniture sales data
plt.figure(figsize=(20, 8))
plt.plot(furniture_office_sales['Order Date'],
         furniture_office_sales['furniture_sales'], 'b-', label='furniture')
plt.plot(furniture_office_sales['Order Date'],
         furniture_office_sales['office_sales'], 'r-', label='office supplies')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Sales of Furniture and Office Supplies')
plt.legend()

# We observe that sales of furniture and office supplies shared a similar seasonal pattern. Early of the year is the off season for both of the two categories. It seems summer time is quiet for office supplies too. in addition, average daily sales for furniture are higher than those of office supplies in most of the months. It is understandable, as the value of furniture should be much higher than those of office supplies. Occasionally, office supplies passed furniture on average daily sales.

#%% Get the first time office supplies’ sales surpassed those of furniture’s.
first_date = furniture_office_sales.ix[np.min(list(np.where(
    furniture_office_sales['office_sales'] > furniture_office_sales['furniture_sales'])[0])), 'Order Date']
print('Office suppliers sales > furniture sales in {0}'.format(first_date.date()))  # 2014-07-01

#%% Time Series Modeling with Prophet
# https://research.fb.com/prophet-forecasting-at-scale/
# Released by Facebook in 2017, forecasting tool Prophet is designed for analyzing time-series that display patterns on different time scales such as yearly, weekly and daily. It also has advanced capabilities for modeling the effects of holidays on a time-series and implementing custom changepoints.
from fbprophet import Prophet

furniture = furniture.rename(
    columns={'Order Date': 'ds', 'Sales': 'y'}
)
furniture_model_prophet = Prophet(interval_width=0.95)
furniture_model_prophet.fit(furniture)

office = office.rename(
    columns={'Order Date': 'ds', 'Sales': 'y'}
)
office_model_prophet = Prophet(interval_width=0.95)
office_model_prophet.fit(office)

furniture_forecast = furniture_model_prophet.make_future_dataframe(
    periods=36, freq='MS')
furniture_forecast = furniture_model_prophet.predict(furniture_forecast)

office_forecast = office_model_prophet.make_future_dataframe(
    periods=36, freq='MS')
office_forecast = office_model_prophet.predict(office_forecast)

#%% Plot for furniture sales
plt.figure(figsize=(18, 6))
furniture_model_prophet.plot(furniture_forecast, xlabel='Date', ylabel='Sales')
plt.title('Furniture Sales')
plt.show()

#%% Plot for office supplies sales
plt.figure(figsize=(18, 6))
office_model_prophet.plot(office_forecast, xlabel = 'Date', ylabel = 'Sales')
plt.title('Office Supplies Sales')

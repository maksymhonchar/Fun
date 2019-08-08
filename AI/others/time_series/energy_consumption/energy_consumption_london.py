#!/usr/bin/env python
# coding: utf-8

# In[1]:


# dataset src: https://data.london.gov.uk/dataset/smartmeter-energy-use-data-in-london-households
# file: UKPN-LCL-smartmeter-sample  (986.99 kB)


# In[2]:


# A Time series is a collection of data points indexed,
# listed or graphed in time order.

# Most commonly, a time series is a sequence taken at
# successive equally spaced points in time.

# Thus it is a sequence of discrete-time data.


# In[3]:


# Load libraries

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np


# In[5]:


raw_data_filename = "UKPN-LCL-smartmeter-sample.csv"

raw_data_df = pd.read_csv(
    raw_data_filename,
    header=0
)


# In[32]:


display(raw_data_df.shape)

display(raw_data_df.head(3))
display(raw_data_df.tail(3))

display(raw_data_df.dtypes)
display(raw_data_df.columns.values)

display(raw_data_df.describe(include='all'))

display(raw_data_df.isnull().sum())

display(raw_data_df[raw_data_df['KWH/hh (per half hour) '] == 'Null'].shape)  # (1, 6)


# In[47]:


raw_date_kwh_df = raw_data_df[['DateTime', 'KWH/hh (per half hour) ']].copy()

raw_date_kwh_df = raw_date_kwh_df.rename(columns={"KWH/hh (per half hour) ": "KWH_hh"})


# In[49]:


# fix row where "KWH_hh" equals 'Null'

display(raw_date_kwh_df[raw_date_kwh_df['KWH_hh'] == 'Null'])  # (1, 6)

raw_date_kwh_df = raw_date_kwh_df.drop([2982])


# In[50]:


# fix dtypes

raw_date_kwh_df.loc[:, 'DateTime'] = pd.to_datetime(raw_date_kwh_df.loc[:, 'DateTime'])

raw_date_kwh_df.loc[:, 'KWH_hh'] = raw_date_kwh_df.loc[:, 'KWH_hh'].astype(float)


# In[51]:


display(raw_date_kwh_df.head())
display(raw_date_kwh_df.dtypes)


# In[60]:


date_kwh_df = raw_date_kwh_df.set_index(raw_date_kwh_df['DateTime'])

date_kwh_df = date_kwh_df.drop(['DateTime'], axis=1)


# In[61]:


display(date_kwh_df.head())


# In[63]:


date_kwh_df.plot(); plt.show()


# In[64]:


# Resampling

# Resampling involves changing the frequency of your
# time series observations.

# One reason why you may be interested in resampling
# your time series data is feature engineering.

# it can be used to provide additional structure or
# insight into the learning problem for supervised learning models.


# In[65]:


weekly = date_kwh_df.resample('W').sum()


# In[68]:


weekly.plot(); plt.show()


# In[72]:


daily = date_kwh_df.resample('D').sum()

daily.plot(); plt.show()


# In[73]:


daily.rolling(30, center=True).sum().plot(); plt.show()


# In[77]:


by_time = date_kwh_df.groupby(date_kwh_df.index.time).mean()  # index == 'DateTime'

hourly_ticks = 4 * 60 * 60 * np.arange(6)

by_time.plot(xticks=hourly_ticks); plt.show()


# In[79]:


pd.plotting.autocorrelation_plot(
    date_kwh_df['KWH_hh']
)

plt.show()


# In[83]:


import fbprophet


# In[82]:


daily_df = daily.copy()
daily_df = daily_df.reset_index()

daily_df = daily_df.rename(
    columns={'DateTime': 'ds', 'KWH_hh': 'y'}
)

display(daily_df.head())


# In[84]:


# In prophet, the changepoint_prior_scale parameter is used
# to control how sensitive the trend is to changes,
# with a higher value being more sensitive and
# a lower value less sensitive.

# https://facebook.github.io/prophet/docs/trend_changepoints.html

prophet_inst = fbprophet.Prophet(changepoint_prior_scale=0.10)
prophet_inst.fit(daily_df)


# In[86]:


forecast = prophet_inst.make_future_dataframe(periods=30*2, freq='D')

forecast = prophet_inst.predict(forecast)


# In[88]:


# The black dots represent the actual values
# the blue line indicates the forecasted values
# the light blue shaded region is the uncertainty.

prophet_inst.plot(forecast)


# In[90]:


# Visualize the overall trend and the component patterns

prophet_inst.plot_components(forecast)


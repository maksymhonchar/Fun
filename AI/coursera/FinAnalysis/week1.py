#%% Load libraries.
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

import pandas as pd
import numpy as np

#%% Load datasets from CSVs.
facebook_data = pd.read_csv('data/facebook.csv')
microsoft_data = pd.read_csv('data/microsoft.csv')

facebook_data = facebook_data.set_index('Date')
microsoft_data = microsoft_data.set_index('Date')

#%% Describe loaded datasets.
display(facebook_data.head(5))
display(microsoft_data.head(5))

display(facebook_data.index.min(), facebook_data.index.max())
display(microsoft_data.index.min(), microsoft_data.index.max())

display(facebook_data.shape)
display(microsoft_data.shape)

display(facebook_data.columns.values)
display(microsoft_data.columns.values)

display(facebook_data.describe())
display(microsoft_data.describe())

#%% Visualize loaded datasets.
fb_2017_2018 = facebook_data.loc['2017-01-01':'2018-12-31'].drop('Volume', axis=1)
ms_2016 = microsoft_data.loc['2016-01-01':'2016-12-31'].drop('Volume', axis=1)

fig, [axis_0, axis_1] = plt.subplots(1, 2, figsize=(15, 5))
axis_0.set_title('Facebook 2017-2018')
fb_2017_2018.plot(ax=axis_0)
axis_1.set_title('Microsoft 2016')
ms_2016.plot(ax=axis_1)
plt.show()

fb_2017_closeyesterday = facebook_data.loc['2017-01-01':'2017-12-31']
fb_2017_closeyesterday = fb_2017_closeyesterday.drop('Volume', axis=1)
fb_2017_closeyesterday['close_yesterday'] = fb_2017_closeyesterday['Close'].shift(1)
fb_2017_closeyesterday.plot()
plt.show()

#%% Add new features to dataset
def fin_df_add_features(df):
    """Add new important features to financial dataset.
    Adding is done inplace.
    """

    # Close price for tomorrow.
    df['close_tomorrow'] = df['Close'].shift(-1)
    # Next day Close price difference.
    df['diff'] = df['close_tomorrow'] - df['Close']
    # Daily return.
    df['return'] = df['diff'] / df['Close']
    # Direction of the Close price.
    df['direction'] = [
        1 if df.loc[el_idx, 'diff'] > 0
        else 0
        for el_idx in df.index
    ]
    # Moving average over 3 days.
    # moving_average_days_cnt = 3
    # df['mov_average_3'] = (df['Close'] + df['Close'].shift(1) + df['Close'].shift(2)) / moving_average_days_cnt 
    df['ma_10'] = df['Close'].rolling(10).mean()
    df['ma_40'] = df['Close'].rolling(40).mean()
    df['ma_50'] = df['Close'].rolling(50).mean()
    df['ma_200'] = df['Close'].rolling(200).mean()

    # todo: dropna for the dataframe?


def plot_custom_close_ma_40_200(df, title=""):
    df['Close'].plot(label='Close price')
    df['ma_10'].plot(label='Moving average 10 days')
    df['ma_40'].plot(label='Moving average 40 days')
    df['ma_50'].plot(label='Moving average 50 days')
    df['ma_200'].plot(label='Moving average 200 days')
    plt.title(title)
    plt.legend()
    plt.show()

facebook_data_ext = facebook_data.copy()
fin_df_add_features(facebook_data_ext)
display(facebook_data_ext.head(3))

microsoft_data_ext = microsoft_data.copy()
fin_df_add_features(microsoft_data_ext)
display(microsoft_data_ext.head(3))

# Plot moving averages
# ToNote: both moving averages SMOOTH the original Close price.
# ma_40 - FAST SIGNAL (MORE CLOSELY associated with Close price) (reflects price over the SHORT history)
# ma_200 - SLOW SIGNAL (reflects price over the LONG history)
plot_custom_close_ma_40_200(facebook_data_ext, 'facebook')
plot_custom_close_ma_40_200(microsoft_data_ext, 'microsoft')

# If ma40 > ma200 -> some traders (trend-following traders) believe the stock price will move upwards for a while.

#%% Build a simple trading strategy.
# Strategy: MA10 > MA50 -> buy and hold one share of stock (long 1 share of stock).
def fin_df_add_simplest_strategy_features(df):
    """Add important features for our simplest strategy"""

    # Denote, whether we long or not.
    df['shares'] = [
        1 if df.loc[el_idx, 'ma_10'] > df.loc[el_idx, 'ma_50']
        else 0
        for el_idx in df.index
    ]
    # Daily profit for our simplest strategy
    df['profit'] = [
        df.loc[el_idx, 'diff'] if df.loc[el_idx, 'shares'] == 1
        else 0
        for el_idx in df.index
    ]
    # Cumulative wealth
    df['wealth'] = df['profit'].cumsum()

facebook_data_trading = facebook_data_ext.drop('Volume', axis=1).copy()
microsoft_data_trading = microsoft_data_ext.drop('Volume', axis=1).copy()

fin_df_add_simplest_strategy_features(facebook_data_trading)
fin_df_add_simplest_strategy_features(microsoft_data_trading)

# Plot all the calculated data
# Note: does not describe 'Volume' values.
fig, [axis_0, axis_1] = plt.subplots(1, 2, figsize=(20, 5))
axis_0.set_title('facebook')
facebook_data_trading.plot(ax=axis_0)
axis_1.set_title('microsoft')
microsoft_data_trading.plot(ax=axis_1)
plt.show()

# Plot the profit data
fig, [axis_0, axis_1] = plt.subplots(1, 2, figsize=(20, 5))
axis_0.set_title('facebook')
facebook_data_trading['profit'].plot(ax=axis_0)
axis_1.set_title('microsoft')
microsoft_data_trading['profit'].plot(ax=axis_1)
plt.show()

# Plot cumulative wealth
fig, [axis_0, axis_1] = plt.subplots(1, 2, figsize=(20, 5))
axis_0.set_title('facebook')
facebook_data_trading['wealth'].plot(ax=axis_0)
axis_1.set_title('microsoft')
microsoft_data_trading['wealth'].plot(ax=axis_1)
plt.show()

#%% 
def print_money_earned_spent(df, name):
    print(name)
    print('Total money you win: {0}'.format(df.loc[df.index[-2], 'wealth']))
    print('Total money you spent: {0}'.format(df.loc[df.index[0], 'Close']))
    print('Final close price: {0}\n'.format(df.loc[df.index[-1], 'Close']))

print_money_earned_spent(facebook_data_trading, 'facebook')
print_money_earned_spent(microsoft_data_trading, 'microsoft')

#%% Questions [markdown]
# Can we find a better signal for trading?
# How do you evaluate your performance of shared strategy correctly?
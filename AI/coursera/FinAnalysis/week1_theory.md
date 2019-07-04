# week1 overview
- import data; pre-process data;
- save financial data;
- manipulate existing data;
- generate new variables using multiple columns;
- build a simple trading strategy based on staying short or long depending on moving averages

# uses
- lower the risk of lending
- predict customers behaviors - recommendation models (across different departments - "customer migrations")

# pandas read_csv vs DataFrame.from_csv
- Both are based on the same underlying function
- read_csv supports more arguments
- it is recommended to use pd.read_csv
    - DataFrame.from_csv exists merely for historical reasons and to keep backwards compatibility.
    - all new features are only added to read_csv
- read_csv is 46x to 490x as fast as DataFrame.from_csv

# Describe dataframe
.head()
.tail()
.describe()
.columns.values
.shape

# select data
.loc
    - df.loc[date1:date2, 'col_name']
.iloc
select column: df['col_name'] df[['col_name_1', 'col_name_2']]

# plot data using pd.DataFrame
.plot()

# New columns for financial dataset
- tomorrow: .shift(-1)
    - note: NaN in the last row
- yesterday: .shift(1)
    - note: NaN in the first row
- priceDiff: price change between tomorrow and today (for 'Close' column)
- return: daily return, priceDiff / Close
- price direction: PriceDiff > 0 ? 1 : 0
- moving average price: either calculate it manually or using a method
    - manually: [-1]+[0]+[1] ... 
    - rolling(n_dats).mean()

# moving average price
- Moving average smooth the original 'Close' price
- MA40 - fast signal; MA200 - slow signal
    - fast signal: reflects the price over a short history
    - slow signal: reflects the price over a long history
- Trend-following traders simplest strategy: if MA40 > MA200 => stock pricie will move upwards for a while, BUY! And vice versa

# Simplest strategy:
- if MA10 is larger than MA50, we will buy and hold 1 share of stock (== we will long one share of stock)
- Calculated profit on facebook and microsoft data: we tend to profit, BUT sometimes we get bankrupt
- Cumulative profit for the whole period: $24

# Questions:
- Can we find a better signal for trading?
- How do you evaluate your performance of shared strategy correctly?
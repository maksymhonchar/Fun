#%% src 1: https://www.tutorialspoint.com/python_pandas/python_pandas_iteration.htm

# In short, basic iteration (for i in object) produces −
# Series − values
# DataFrame − column labels
# Panel − item labels

# Note − Do not try to modify any object while iterating.
# Iterating is meant for reading and the iterator returns a
#   copy of the original object (a view), thus the changes
#   will not reflect on the original object.

#%% Iterating a dataframe

import pandas as pd
import numpy as np

#%% Define dataframe
N = 20

df = pd.DataFrame({
   'A': pd.date_range(start='2016-01-01',periods=N,freq='D'),
   'x': np.linspace(0,stop=N-1,num=N),
   'y': np.random.rand(N),
   'C': np.random.choice(['Low','Medium','High'],N).tolist(),
   'D': np.random.normal(100, 10, size=(N)).tolist()
})

display(df)

#%% for-in iteration == columns
for col in df:
    print(col)  # A x y C D ; note: ordered

#%% Alternatives

# iteritems() - (k,v) pairs
# iterrows() - (idx, series) pairs
# itertuples() - rows as namedtuples

#%% iteritems()
for k, v in df.iteritems():
    print(k, v)  # (str, pd.Series)

#%% iterrows()
for idx, row in df.iterrows():
    print(type(idx), type(row))  # (int, pd.Series)

# Note − Because iterrows() iterate over the rows, it doesn't preserve the data type across the row.

#%% itertuples()
for row in df.itertuples():
    print(type(row))  # <class 'pandas.core.frame.Pandas'>

#%%

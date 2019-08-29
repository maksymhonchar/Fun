#%% Import libraries

import pandas as pd


#%% Load dataset

train_df = pd.read_csv('/home/max/Downloads/titanic/train.csv')
test_df = pd.read_csv('/home/max/Downloads/titanic/test.csv')

concat_df = pd.concat([train_df, test_df], ignore_index=False, sort=False)

display(concat_df.shape)


#%% Approach 1: iterrows
def test_iterrows(df):
    for i,row in df.iterrows():
        val = row['Age']

%timeit test_iterrows(concat_df)  # 125 ms 


#%% Approach 2: .loc, .iloc
def test_loc_iloc(df):
    for i in df.index:
        val = df.loc[i, 'Age']

%timeit test_loc_iloc(concat_df)  # 600 ms


#%% Approach 3: .get_value, .set_value

# note: deprecated 

def test_getval_setval(df):
    for i in df.index:
        val = df.get_value(i, 'Age')

%timeit test_getval_setval(concat_df)  # 23.5ms


#%% Approach 3.2: .at / .iat
def test_at_iat(df):
    for i in df.index:
        val = df.at[i, 'Age']

%timeit test_at_iat(concat_df)  # 24.8ms


#%% Approach 4: .apply()

%timeit concat_df['Age'].apply(lambda x: x)  # 412 us

#%% Conclusion
# apply > at/iat > loc > iterrows
# src
https://towardsdatascience.com/different-ways-to-iterate-over-rows-in-a-pandas-dataframe-performance-comparison-dc0d5dcef8fe

# functions to compare:
1. use_column: use pandas column operation
2. use_panda_apply: use pandas apply function
3. use_for_loop_loc: uses the pandas loc function
4. use_for_loop_at: use the pandas at function(a function for accessing a single value)
5. use_for_loop_iat: use the pandas iat function(a function for accessing a single value)
6. use_numpy_for_loop: get the underlying numpy array from column, iterate , compute and assign the values as a new column to the dataframe
7. use_iterrows: use pandas iterrows function to get the iterables to iterate
8. use_zip: use python built-in zip function to iterate, store results in a numpy array then assign the values as a new column to the dataframe upon completion

# Conclusions:
- Column operation and apply are both relatively fast
- Select using at() and iat() is faster than loc()
- Location-based indexing of numpy array is faster than locating-based indexing on a pandas dataframe
    - == Consider extracting the underlying values as a numpy array then perform the processing/analysing

- zip() is relatively fast for small dataset - even faster than apply() for N < 1000
- iat() and at() indexing can be 30 times faster than loc()
    - avoid using loc() for updating or access single value, use iat() and at() instead

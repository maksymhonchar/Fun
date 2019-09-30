#!/usr/bin/env python
# coding: utf-8

# In[1]:


# src

# https://medium.com/journey-2-artificial-intelligence/lda-linear-discriminant-analysis-using-python-2155cf5b6398

# https://github.com/sambit9238/DataScience/blob/master/LDA.ipynb?source=post_page-----2155cf5b6398----------------------


# In[2]:


# LDA is a supervised dimensionality reduction technique
# The goal is to project a dataset onto a lower-dimensional space with good class-separability in order avoid overfitting (“curse of dimensionality”) and also reduce computational costs

# Basically, the added advantage LDA gives over PCA is to tackle overfitting.

# The general LDA approach is very similar to a Principal Component Analysis.
# But in addition to finding the component axes that maximize the variance of our data (PCA), we are additionally interested in the axes that maximize the separation between multiple classes (LDA).

# Steps of LDA:
# Compute d-dimensional mean vectors for different classes from the dataset, where d is the dimension of feature space.
# Compute in-between class and with-in class scatter matrices.
# Compute eigen vectors and corresponding eigen values for the scatter matrices.
# Choose k eigen vectors corresponding to top k eigen values to form a transformation matrix of dimension d x k.
# Transform the d-dimensional feature space X to k-dimensional feature space X_lda via the transformation matrix.


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

import pandas as pd

import scipy.stats as sstats


# In[4]:



CSV_PATH = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt'
main_df = pd.read_csv(
    CSV_PATH,
    names=['var', 'skewness', 'curtosis', 'entropy', 'class'],
    index_col=False
)


# In[5]:


display( main_df.shape )
display( main_df.sample() )
display( main_df.isnull().sum() )
display( main_df.duplicated().sum() )

display( main_df['class'].value_counts() )

display( main_df.describe(include='all') )

main_df.info()


# In[6]:


# If the K-S statistic is small or the p-value is high, then
# we cannot reject the hypothesis that the distributions of the two samples
# are the same.

fig, ax = plt.subplots( 1, 4, figsize=(15, 5) )
for idx, col_name in enumerate( main_df.columns[:-1] ):  # skip 'class' column
    sns.distplot( main_df[col_name], ax=ax[idx] )
    print('{0}: skew:{1}, kurt:{2}, KS-score:{3}\n'.format(
        col_name,
        main_df[col_name].skew(),
        main_df[col_name].kurt(),
        sstats.kstest( main_df[col_name], 'norm' )
    ))
plt.show()


# In[7]:


sns.pairplot(
    main_df,
    hue='class'
)


# In[8]:


# Compute the 4-dimensional mean vectors for both the classes
# Unlike PCA, standardization of the data is not needed in LDA as it doesn't affect the output.

mean_vec = []

for unique_class_value in main_df['class'].unique():
    mean_vec.append(
        np.array( main_df[ main_df['class'] == 0 ].mean()[:4] )
    )


# In[9]:


# Calculate:
# 1. the with-in class scatter matrices
# 2. in-between class scatter matrices

# With-in class scatter matrices
SW = np.zeros( (4, 4) )

for i in range(2):  # for each unique class value
    per_class_sc_mat = np.zeros( (4, 4) )
    for j in range( main_df[main_df["class"] == i].shape[0] ):
        row = main_df.loc[j][:4].values.reshape(4,1)
        mv = mean_vec[i].reshape(4,1)
        per_class_sc_mat += (row - mv).dot( (row - mv).T )
    SW += per_class_sc_mat


# In-between class scatter matrices
overall_mean = np.array(main_df.drop("class", axis=1).mean())

SB = np.zeros( (4, 4) )

for i in range(2):
    n = main_df[main_df["class"]==i].shape[0]
    mv = mean_vec[i].reshape(4,1)
    overall_mean = overall_mean.reshape(4,1) # make column vector
    SB += n * (mv - overall_mean).dot((mv - overall_mean).T)


# In[10]:


# Solve the generalized eigenvalue problem to obtain the linear discrirminants

e_vals, e_vecs = np.linalg.eig(  # eigenvalues and eigenvectors
    np.linalg.inv( SW ).dot( SB )
)


# In[11]:


# Make a list of (eigenvalue, eigenvector) tuples
# Sort the tuples from high to low values

e_pairs = [
    ( np.abs(e_vals[i]), e_vecs[:,i] )
    for i in range( len(e_vals) )
]

e_pairs.sort( reverse=True )


# In[12]:


# Select top k eigenvectors corresponding to top k eigenvalues

# For data compression purpose, we generally go for 99% variance retention, while for visualization we make the dimension to 2 or 3.

# Here, we till take top-2 eigen values corresponding eigen vectors for visualization purpose.
# But we will the eigen vector belongs to largest eigen value retains nearly 100% variance, so we can discard other 3 too.

W = np.hstack((
    e_pairs[0][1].reshape(4,1),
    e_pairs[1][1].reshape(4,1)
))


# In[13]:


# Transform the 4-dim feature space to 2-dim feature subspace

X = main_df.iloc[:,0:4].values
X_lda = X.dot(W)

main_df["PC1"] = X_lda[:,0]
main_df["PC2"] = X_lda[:,1]


# In[14]:


# Visualize 2 new components

sns.scatterplot(
    x='PC1', y='PC2',
    hue='class',
    data=main_df,
)
plt.show()


# In[15]:


sns.pairplot(
    main_df[ ['PC1', 'PC2', 'class'] ],
    hue='class'
)


# In[16]:


# sklearn and LDA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

X = main_df.iloc[:,0:4].values
y = main_df.iloc[:,4].values

model_lda = LDA( n_components=2 )

X_lda_skl = model_lda.fit_transform( X, y )
main_df['skl_PC1'] = X_lda_skl[:, 0]


# In[17]:


# Visualize results for LDA from sklearn

sns.scatterplot(
    x='skl_PC1', y='class',
    data=main_df
)


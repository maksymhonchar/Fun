#!/usr/bin/env python
# coding: utf-8

# In[16]:

# fifa data src: https://www.kaggle.com/mathan/fifa-2018-match-statistics


# Load libraries

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd

# factor_analyzer documentation
# https://buildmedia.readthedocs.org/media/pdf/factor-analyzer/latest/factor-analyzer.pdf
from factor_analyzer.factor_analyzer import calculate_kmo
from factor_analyzer import FactorAnalyzer

from sklearn.decomposition import FactorAnalysis


# In[2]:


# Load data

MAIN_DATASET_FILEPATH = 'data/FIFA 2018 Statistics.csv'
fifa_df = pd.read_csv( MAIN_DATASET_FILEPATH, header=0 )


# In[3]:


# fifa_df.info()


# In[4]:


# fifa_df.isnull().sum()


# In[5]:


# Get only numeric columns

fifa_num = fifa_df.iloc[:, 3:20]
fifa_num['first_goal'] = fifa_df['1st Goal']


# In[6]:


# Some rows have first_goal=nan - replace nan values with '0' value

fifa_num['first_goal'] = fifa_num['first_goal'].fillna( 0 )


# In[7]:


# Factor Analysis

# Step 1. Identify relevant variables - Kaiser-Meyer-Olkin (KMO) Test
# This test is a measure of how suited your data is for Factor Analysis

# KMO returns values between 0 and 1.
# A rule of thumb for interpreting the statistic:
# KMO=[0.8; 1] indicate the sampling is adequate.
# KMO=[0.5; 0.6] indicate the sampling is not adequate and that remedial
# action should be taken.
# KMO -> 0 means that there are large partial correlations compared to the
# sum of correlations.
# In other words, there are widespread correlations which are a large problem
# for factor analysis.

# 0.00 to 0.49 unacceptable.
# 0.50 to 0.59 miserable.
# 0.60 to 0.69 mediocre.
# 0.70 to 0.79 middling.
# 0.80 to 0.89 meritorious.
# 0.90 to 1.00 marvelous

kmo_all, kmo_model = calculate_kmo( fifa_num )

display( kmo_model )  # kmo_model = 0.5555331829032943 -> miserable == inadequate data

col_msa_value_df = pd.DataFrame({
    'col_name': fifa_num.columns,
    'MSA_value': kmo_all  # measure of sampling adequacy
})

display(col_msa_value_df)


# In[8]:


# MSA < 0.5 - variable should be dropped
# MSA > 0.6 are suitable
# MSA > 0.8 are very well suited for factor analysis

columns_to_remove = col_msa_value_df[ col_msa_value_df['MSA_value'] < 0.5 ]['col_name'].values


# In[9]:


# Drop columns with low MSA

fifa_num = fifa_num.drop( columns_to_remove, axis=1 )

# display( fifa_num )


# In[10]:


# Step 2. Overview correlation matrix for numerical columns with MSE>0.5

plt.figure( figsize=(10, 10) )
sns.heatmap(
    fifa_num.corr(),
    annot=True
)
plt.autoscale()
plt.show()


# In[27]:


# Step 3. Perform factor analysis

# Choose the number of factors
transformer = FactorAnalysis( n_components=4 )
X_transformed = transformer.fit_transform(fifa_num) 

display(X_transformed.shape, fifa_num.shape)

plt.figure( figsize=(10, 10) )
sns.heatmap(
    transformer.get_covariance(),
    annot=True
)
plt.autoscale()
plt.show()


# In[ ]:


##########################################################################


# In[ ]:


# Factor analysis is regression method used to explore datasets to find root causes that
# explain why data is acting a certain way.

# Example: in marketing analysis, identify useful customer segments for further use
# apply factor analysis as simple way to group respondents on your survey into meaningful
# customer segments based on similarities in how response tends to apply user to certain
# category.

# Factors (aka latent variables) = Variables that are quite meaningful but are inferred
# and not directly observable.

# Factor analysis ASSUMPTIONS
# features are metrics
# features are continuous or ordinal
# there is r>0.3 correlation between the faetures in the dataset
# >100 observations and >5 observations per features
# sample is homogenous


# In[28]:


from sklearn.datasets import load_iris
from sklearn.decomposition import FactorAnalysis


# In[42]:


iris = load_iris()

iris_df = pd.DataFrame(
    iris.data, columns=iris.feature_names
)
iris_df['target'] = iris.target

display( iris_df.sample() )


# In[44]:


# Corr: number of features with corr > 0.3

figure = plt.figure( figsize=(10, 10) )
sns.heatmap(
    iris_df.corr(),
    annot=True
)
plt.autoscale()
plt.show()

# all 4 are > 0.3


# In[48]:


# Fitting FactorAnalysis model in order to reduce the dataset dimensionality by
# encoding features that contain the most information == contain the most variance in
# the dataset == latent variables

fa = FactorAnalysis(n_components=2)
transformed_fa = fa.fit_transform(iris.data)

display(transformed_fa.shape, iris.data.shape)
display(transformed_fa[:5])


# In[56]:


# Plot these 2 features

transformed_iris_df = pd.DataFrame(
    transformed_fa, columns=['feature1', 'feature2']
)
transformed_iris_df['target'] = iris.target

plt.scatter(
    x=transformed_iris_df['feature1'], y=transformed_iris_df['feature2'],
    c=transformed_iris_df['target']
)
plt.show()


# In[57]:


# Important distinction between PCA and factor analysis

# PCA - linear transformation where the 'first' component 'explains' the variance of the data,
# and each subsequent component is orthogonal to the first component
# SO, PCA - take dataset of N dimensions and go down to some space of M dims, M<N

# Factor Analysis - works under the assumption that there are only M important feautres,
# and a linear combination of these features + noise craetes the dataset in N dimensions.


#!/usr/bin/env python
# coding: utf-8

# In[1]:


# src: https://pythonfordatascience.org/anova-python/

# The analysis of variance (ANOVA) can be thought of as an extension to the t-test

# The independent t-test is used to compare the means of a condition between 2 groups.
# ANOVA is used when one wants to compare the means of a condition between 2+ groups.

# ANOVA is an omnibus test, meaning it tests the data as a whole
# ANOVA tests if there is a difference in the mean somewhere in the
# model (testing if there was an overall effect), but it does not tell one
# where the difference is if the there is one.

# ANOVA - mathematically speaking, it’s more of a regression model and is
# considered a generalized linear model (GLM).

# regression equation: outcomei = (model) + errori
# with N groups: outcomei = b0 + b1Group1 + b2Group1 + errori

# ANOVA assumptions == same for linear regression:
# Normality
    # Caveat to this is, if group sizes are equal, the F-statistic
    # is robust to violations of normality
# Homogeneity of variance
    # Same caveat as above, if group sizes are equal, the F-statistic is robust
    # to this violation
# Independent observations

# If these assumptions are not met, and one does not want to transform the data,
# an alternative test that could be used is the Kruskal-Wallis H-test or Welch’s ANOVA.


# In[2]:


# Load libraries
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

import pandas as pd

import scipy.stats as stats

import statsmodels.api as sm
from statsmodels.formula.api import ols

import researchpy as rp


# In[3]:


# Load the data

raw_df = pd.read_csv("https://raw.githubusercontent.com/Opensourcefordatascience/Data-sets/master/difficile.csv")


# In[4]:


# Explore the data

display(raw_df.head(3), raw_df.tail(3))

display(raw_df.shape)

display(raw_df.columns.values)

display([pd.unique(raw_df[col_name]) for col_name in raw_df.columns.values])

display(raw_df.describe(include='all'))


# In[5]:


# Initial work on raw_df

main_df = raw_df.copy()

# Drop 'person' column - we don't need it
main_df = main_df.drop('person', axis=1)

# Map 'dose' column values with string analogues
main_df['dose'] = main_df['dose'].map({1: 'placebo', 2: 'low', 3: 'high'})

display(main_df['dose'])


# In[6]:


display(rp.summary_cat(main_df['dose']))

display(rp.summary_cat(main_df['libido']))


# In[7]:


rp.summary_cont(main_df['libido'].groupby(main_df['dose']))


# In[8]:


# ANOVA example with scipy.stats

display(stats.f_oneway(
    main_df['libido'][main_df['dose'] == 'high'],  # sample1
    main_df['libido'][main_df['dose'] == 'low'],  # sample2
    main_df['libido'][main_df['dose'] == 'placebo']  # sample3
))


# In[9]:


# ANOVA with statsmodels

results = ols('libido ~ C(dose)', data=main_df).fit()
results.summary()


# In[10]:


# src2: https://www.marsja.se/four-ways-to-conduct-one-way-anovas-using-python/


# plant_dataset_df = pd.read_csv('PlantGrowth.csv')
# 
# display(plant_dataset_df.head(3))
# 
# display(plant_dataset_df.describe().T)
# 
# display(rp.summary_cat(plant_dataset_df['group']))
# 
# display(rp.summary_cont(plant_dataset_df['weight']))

# In[22]:


plant_dataset_df.boxplot('weight', by='group', figsize=(12, 8))
plt.show()


# In[26]:


mod = ols('weight ~ group', data=plant_dataset_df).fit()
                
aov_table = sm.stats.anova_lm(mod, typ=2)
display(aov_table)


# In[29]:


# py3 - not working
# py2 output - "Anova: Single Factor on weight" neat table

# from pyvttbl import DataFrame

# aov_pyvttbl = plant_dataset_df.anova1way('weight', 'group')
# display(aov_pyvttbl)


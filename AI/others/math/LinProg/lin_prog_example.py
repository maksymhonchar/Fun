#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# src: https://www.kaggle.com/mchirico/linear-programming


# In[9]:


import matplotlib.pyplot as plt


# In[1]:


# scipy.linprog 
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html


# In[2]:


# Consider the following problem:
# Minimize: f = -1x[0] + 4x[1]
# Subject to:
# -3x[0] + 1x[1] <= 6
# 1x[0] + 2x[1] <= 4
# x[1] >= -3
# -inf <= x[0] <= +inf

c = [-1, 4]  # minimize function
A = [ [-3, 1], [1, 2] ]  # subject to 1, 2
b = [6, 4]  # subject to 1, 2 (free term)

x0_bounds = (None, None)
x1_bounds = (-3, None)


# In[7]:


from scipy.optimize import linprog

result = linprog(
    c,  # coeffs of the linear objective function to be minimized
    A_ub=A,  # inequality constraint matrix: coeffs of a linear inequality
    b_ub=b,  # inequality constraint vector: upper bound A_ub @ x
    bounds=(x0_bounds, x1_bounds),  # sequence of (min, max) pairs for each element in x
    options={'disp': True}
)

display(result)


# In[14]:


# Example 2

# A trading company is looking for a way to maximize profit per transportation of their goods.
# The company has a train available with 3 wagons.
# When stocking the wagons they can choose between 4 types of cargo, each with its own specifications.
# How much of each cargo type should be loaded on which wagon in order to maximize profit?

data_matrix = [['Train Wagon', 'Item Capacity', 'Space Capacity'],
               ['w1', 10, 5000],
               ['w2', 8, 4000],
               ['w3', 12, 8000],]

data_matrix_2 = [['Cargo<br>Type', '#Items Available', 'Volume','Profit'],
               ['c1', 18, 400,2000],
               ['c2', 10, 300,2500],
               ['c3', 5, 200,5000],
               ['c4', 20, 500,3500]]

# Objective function
# max: +2000 C1 +2500 C2 +5000 C3 +3500 C4 +2000 C5 +2500 C6 +5000 C7 +3500 C8 +2000 C9 +2500 C10 +5000 C11 +3500 C12;
# Flip sign above to get MIN PROBLEM

# Constraints
# +C1 +C2 +C3 +C4 <= 10;
# +C5 +C6 +C7 +C8 <= 8;
# +C9 +C10 +C11 +C12 <= 12;
# +400 C1 +300 C2 +200 C3 +500 C4 <= 5000;
# +400 C5 +300 C6 +200 C7 +500 C8 <= 4000;
# +400 C9 +300 C10 +200 C11 +500 C12 <= 8000;
# +C1 +C5 +C9 <= 18;
# +C2 +C6 +C10 <= 10;
# +C3 +C7 +C11 <= 5;
# +C4 +C8 +C12 <= 20;

# What if we get rid of item constraint?
# Change min to max
c = [-2000,-2500,-5000,-3500,-2000,-2500,-5000,-3500,-2000,-2500,-5000,-3500]
xb=[]
for i in range(0,12):
    xb.append((0, None))

A = [
     [400,300,200,500,0,0,0,0,0,0,0,0,],
     [0,0,0,0,400,300,200,500,0,0,0,0,],
     [0,0,0,0,0,0,0,0,400,300,200,500],
     [1,0,0,0,1,0,0,0,1,0,0,0],
     [0,1,0,0,0,1,0,0,0,1,0,0],
     [0,0,1,0,0,0,1,0,0,0,1,0],
     [0,0,0,1,0,0,0,1,0,0,0,1],
    ]    

b = [5000,4000,8000,18,10,5,20]

res = linprog(c, A_ub=A, b_ub=b, bounds=xb,
              options={"disp": True})
print(res)


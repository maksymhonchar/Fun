#!/usr/bin/env python
# coding: utf-8

# # 100 numpy exercises
# 
# This is a collection of exercises that have been collected in the numpy mailing list, on stack overflow and in the numpy documentation. The goal of this collection is to offer a quick reference for both old and new users but also to provide a set of exercises for those who teach.
# 
# 
# If you find an error or think you've a better way to solve some of them, feel free to open an issue at <https://github.com/rougier/numpy-100>

# #### 1. Import the numpy package under the name `np` (★☆☆)

# In[1]:


import numpy as np


# #### 2. Print the numpy version and the configuration (★☆☆)

# In[2]:


print(np.version.version)

np.__config__.show()


# #### 3. Create a null vector of size 10 (★☆☆)

# In[3]:


v = np.zeros(10)


# #### 4.  How to find the memory size of any array (★☆☆)

# In[4]:


# Way 1
import sys
v = np.ones(10)
v_memory_size = v.size * sys.getsizeof(v[0])  # bytes

print(sys.getsizeof(v))  # 176
print(sys.getsizeof(v[0]))  # 32
print(v_memory_size)  # 320

# Way 2
v = np.ones(10)
v_memory_size = v.size * v.itemsize

print(v_memory_size)  # 80  # bytes


# #### 5.  How to get the documentation of the numpy add function from the command line? (★☆☆)

# In[5]:


print(help(np.add))


# #### 6.  Create a null vector of size 10 but the fifth value which is 1 (★☆☆)

# In[6]:


v = np.zeros(10)
v[4] = 1

display(v)


# #### 7.  Create a vector with values ranging from 10 to 49 (★☆☆)

# In[7]:


v = np.arange(10, 50)

display(v)


# #### 8.  Reverse a vector (first element becomes last) (★☆☆)

# In[8]:


v = np.arange(10)
v = np.flip(v)

display(v)


# #### 9.  Create a 3x3 matrix with values ranging from 0 to 8 (★☆☆)

# In[9]:


m = np.arange(0, 9).reshape(3, 3)

display(m)


# #### 10. Find indices of non-zero elements from \[1,2,0,0,4,0\] (★☆☆)

# In[10]:


v = np.asarray([1, 2, 0, 0, 4, 0])
nonzero_elems = np.nonzero(v)

display(nonzero_elems)


# #### 11. Create a 3x3 identity matrix (★☆☆)

# In[11]:


m = np.identity(3)

display(m)


# #### 12. Create a 3x3x3 array with random values (★☆☆)

# In[12]:


m = np.random.randint(0, 10, size=(3, 3, 3))

display(m)


# #### 13. Create a 10x10 array with random values and find the minimum and maximum values (★☆☆)

# In[13]:


m = np.random.randint(0, 100, size=(10, 10))
min_value = np.min(m)
max_value = np.max(m)

display(m)
display(min_value, max_value)


# #### 14. Create a random vector of size 30 and find the mean value (★☆☆)

# In[14]:


v = np.random.randint(0, 10, 30)
mean_v = np.mean(v)

display(v)
display(mean_v)


# #### 15. Create a 2d array with 1 on the border and 0 inside (★☆☆)

# In[15]:


# Way 1
v = np.zeros((5, 5))
v = np.pad(v, pad_width=1, mode='constant', constant_values=1)

display(v)

# Way 2
v = np.ones((10, 10))
v[1:-1, 1:-1] = 0

display(v)


# #### 16. How to add a border (filled with 0's) around an existing array? (★☆☆)

# In[16]:


# Way 1
v = np.ones((5, 5))
v = np.pad(v, pad_width=1, mode='constant', constant_values=0)

display(v)

# Way 2
v = np.random.randint(0, 10, size=(5, ))
v = np.pad(v, pad_width=1, mode='constant', constant_values=0)

display(v)


# #### 17. What is the result of the following expression? (★☆☆)

# ```python
# 0 * np.nan
# np.nan == np.nan
# np.inf > np.nan
# np.nan - np.nan
# np.nan in set([np.nan])
# 0.3 == 3 * 0.1
# ```

# In[17]:


print(0 * np.nan)  # np.nan
print(np.nan == np.nan)  # False
print(np.inf > np.nan)  # False
print(np.nan - np.nan)  # np.nan
print(np.nan in set([np.nan]))  # True
print(0.3 == 3 * 0.1)  # False


# #### 18. Create a 5x5 matrix with values 1,2,3,4 just below the diagonal (★☆☆)

# In[18]:


# Way 1
m = np.zeros((5, 5))
rng = [1, 2, 3, 4,]
for i in rng:
    m[i, i-1] = 1

display(m)

# Way 2
m = np.diag(np.arange(1, 5), k=-1)

display(m)


# #### 19. Create a 8x8 matrix and fill it with a checkerboard pattern (★☆☆)

# In[19]:


m = np.zeros((8, 8))
m[1::2, 1::2] = 1
m[0::2, 0::2] = 1

display(m)


# #### 20. Consider a (6,7,8) shape array, what is the index (x,y,z) of the 100th element?

# In[20]:


idx = np.unravel_index(99, (6, 7, 8))

display(idx)


# #### 21. Create a checkerboard 8x8 matrix using the tile function (★☆☆)

# In[21]:


checkboard = np.tile([[0, 1], [1, 0]], (4, 4))

display(checkboard)


# #### 22. Normalize a 5x5 random matrix (★☆☆)

# In[22]:


m = np.random.randint(1, 10, size=(5, 5))
m_max, m_min = m.max(), m.min()
m_norm = (m - m_min) / (m_max - m_min)

display(m_norm)


# #### 23. Create a custom dtype that describes a color as four unsigned bytes (RGBA) (★☆☆)

# In[23]:


rgba_dtype = np.dtype([
    ('red', np.unsignedinteger),
    ('green', np.unsignedinteger),
    ('blue', np.unsignedinteger),
    ('alpha', np.unsignedinteger)
])
rgba_val = np.array(
    [(1, 2, 3, 4), (2, 3, 4, 5), (3, 4, 5, 6)],
    dtype=rgba_dtype
)

display(rgba_dtype)
display(rgba_val)


# #### 24. Multiply a 5x3 matrix by a 3x2 matrix (real matrix product) (★☆☆)

# In[24]:


m_1 = np.random.randint(1, 10, size=(5, 3))
m_2 = np.random.randint(1, 10, size=(3, 2))
mult = np.dot(m_1, m_2)

display(mult)


# #### 25. Given a 1D array, negate all elements which are between 3 and 8, in place. (★☆☆)

# In[25]:


v = np.random.randint(1, 10, size=(5, ))
neg_val_betw_3_8 = lambda val: -val if 3 < val < 8 else val
vect_neg_val_betw_3_8 = np.vectorize(neg_val_betw_3_8)

display(vect_neg_val_betw_3_8(v))


# #### 26. What is the output of the following script? (★☆☆)

# ```python
# # Author: Jake VanderPlas
# 
# print(sum(range(5),-1))
# from numpy import *
# print(sum(range(5),-1))
# ```

# In[26]:


print(sum(range(5), -1))

print(np.sum(range(5), -1))

display(list(range(5)))

display(np.sum(range(5), -1))


# #### 27. Consider an integer vector Z, which of these expressions are legal? (★☆☆)

# ```python
# Z**Z
# 2 << Z >> 2
# Z <- Z
# 1j*Z
# Z/1/1
# Z<Z>Z
# ```

# In[27]:


Z = np.arange(10)

display(Z**Z)  # OK
display(2 << Z >> 2)  # OK

# display(Z.tolist())
# display((2 << Z).tolist())

display(Z <- Z)  # OK

# display(Z < -Z)

display(Z / 1 / 1)  # OK

# display(Z < Z > Z)  # NOT OK


# #### 28. What are the result of the following expressions?

# ```python
# np.array(0) / np.array(0)
# np.array(0) // np.array(0)
# np.array([np.nan]).astype(int).astype(float)
# ```

# In[28]:


display(np.array(0) / np.array(0))  # nan + warning

display(np.array(0) // np.array(0))  # 0 + warning

display(np.array([np.nan]).astype(int).astype(float))  # weird value


# #### 29. How to round away from zero a float array ? (★☆☆)

# In[29]:


v = np.random.uniform(-10, 10, 10)
v_rounded_away = np.copysign(np.ceil(np.abs(v)), v)

display(v)
display(v_rounded_away)


# #### 30. How to find common values between two arrays? (★☆☆)

# In[30]:


v_1 = np.random.randint(1, 10, (10, ))
v_2 = np.random.randint(1, 10, (10, ))
common_values = np.intersect1d(v_1, v_2)

display(v_1.tolist(), v_2.tolist())
display(common_values.tolist())


# 
# #### 31. How to ignore all numpy warnings (not recommended)? (★☆☆)

# In[31]:


# Way 1: numpy warnings
defaults = np.seterr(all="ignore")

# Way 2: all warnings
import warnings
warnings.filterwarnings('ignore')

display(np.array(0) / np.array(0))  # nan, no warnings


# #### 32. Is the following expressions true? (★☆☆)

# ```python
# np.sqrt(-1) == np.emath.sqrt(-1)
# ```

# In[32]:


display(np.sqrt(-1) == np.emath.sqrt(-1))

display(np.sqrt(-1))  # nan
display(np.emath.sqrt(-1), type(np.emath.sqrt(-1)))  # 1j, numpy.complex128


# #### 33. How to get the dates of yesterday, today and tomorrow? (★☆☆)

# In[33]:


from datetime import datetime

day_timedelta = np.timedelta64(1, 'D')

# today = np.datetime64(datetime.now(), 'D')
today = np.datetime64('today', 'D')
yesterday = today - day_timedelta
tomorrow = today + day_timedelta

display(today, yesterday, tomorrow)


# #### 34. How to get all the dates corresponding to the month of July 2016? (★★☆)

# In[34]:


jul_2016_days = np.arange('2016-07', '2016-08', dtype='datetime64[D]')

display(jul_2016_days)


# #### 35. How to compute ((A+B)\*(-A/2)) in place (without copy)? (★★☆)

# In[35]:


A = 5
B = 10
minAdiv2 = np.ndarray(1)
AplusB = np.ndarray(1)
final_mult = np.ndarray(1)

np.add(A, B, out=AplusB)
np.divide(A, 2, out=minAdiv2)
np.negative(minAdiv2, out=minAdiv2)
np.multiply(AplusB, minAdiv2, out=final_mult)

display(minAdiv2, AplusB, final_mult)


# #### 36. Extract the integer part of a random array using 5 different methods (★★☆)

# In[36]:


v = np.random.rand(10) * np.random.randint(20)

display(v)

# display([int(item) for item in v])
display(v.astype(int))

# display([int(np.floor(item)) for item in v])
display(np.floor(v))

display(np.trunc(v))
display(v - v % 1)
display(np.ceil(v) - 1)


# #### 37. Create a 5x5 matrix with row values ranging from 0 to 4 (★★☆)

# In[37]:


# Way 1
m = np.tile(np.arange(5), 5).reshape(5, 5)

display(m)

# Way 2
m = np.zeros((5, 5))
m += np.arange(5)

display(m)


# #### 38. Consider a generator function that generates 10 integers and use it to build an array (★☆☆)

# In[38]:


# yield np.random.randint(1, 100, (1, 10))

def ten_int_arr():
    for i in range(10):
        yield i
    
v = np.fromiter(ten_int_arr(), dtype=float, count=-1)

display(v)


# #### 39. Create a vector of size 10 with values ranging from 0 to 1, both excluded (★★☆)

# In[39]:


v = np.linspace(0, 1, 11, endpoint=False)[1:]

display(v)


# #### 40. Create a random vector of size 10 and sort it (★★☆)

# In[40]:


v = np.random.rand(10)
v = np.sort(v)
# %timeit v.sort()  # quicksort  # 63.5us 10000elems
# %timeit v.sort(kind='mergesort')  # 99.5us 10000elems
# %timeit v.sort(kind='heapsort')  # 588us 10000elems

display(v)


# #### 41. How to sum a small array faster than np.sum? (★★☆)

# In[41]:


v = np.linspace(0, 1, 100)

get_ipython().run_line_magic('timeit', 'np.sum(v)  # 3.02 us')

from functools import reduce
sum_reduce = lambda x, y: x + y
get_ipython().run_line_magic('timeit', 'reduce(sum_reduce, v)  # 21.4us')

get_ipython().run_line_magic('timeit', 'np.add.reduce(v)  # 1.24us')


# #### 42. Consider two random array A and B, check if they are equal (★★☆)

# In[42]:


A = np.arange(1000000)
B = np.arange(1000000)

get_ipython().run_line_magic('timeit', 'all(np.intersect1d(A, B) == A)  # 262ms')

get_ipython().run_line_magic('timeit', 'all(A == B)  # 14.7ms')

get_ipython().run_line_magic('timeit', 'np.all(A == B)  # 1.82us')

get_ipython().run_line_magic('timeit', 'np.array_equal(A, B)  # 1.98ms')

get_ipython().run_line_magic('timeit', 'np.allclose(A, B)  # 18.8ms')


# #### 43. Make an array immutable (read-only) (★★☆)

# In[43]:


v = np.arange(1, 10)

# Way 1
v.setflags(write=False)

# Way 2
v.flags.writeable = False

display(v.flags)  # type(v.flags) == numpy.flagsobj


# #### 44. Consider a random 10x2 matrix representing cartesian coordinates, convert them to polar coordinates (★★☆)

# In[55]:


cartesian_coords = np.random.rand(10, 2)  # (x, y)

# Way 1 - incorrect todo:
polar_r = lambda row: np.sqrt( row[0]**2 + row[1]**2 )
polar_theta = lambda row: np.arctan2(row[1], row[0])
polar_r_vals = np.array(list(map(polar_r, cartesian_coords)))
polar_theta_vals = np.array(list(map(polar_theta, cartesian_coords)))

display(polar_r_vals)
display(polar_theta_vals)

# Way 2
X, Y = cartesian_coords[:, 0], cartesian_coords[:, 1]
R = np.sqrt(X**2 + Y**2)
T = np.arctan2(Y, X)

display(R, T)


# #### 45. Create random vector of size 10 and replace the maximum value by 0 (★★☆)

# In[58]:


v = np.random.rand(10)
max_elem_idx = np.argmax(v)

display(v)

v[max_elem_idx] = 0

display(v)


# #### 46. Create a structured array with `x` and `y` coordinates covering the \[0,1\]x\[0,1\] area (★★☆)

# In[86]:


N = 3
m = np.zeros((N, N), [('x', float), ('y', float)])
m['x'], m['y'] = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N))

display(m)


# ####  47. Given two arrays, X and Y, construct the Cauchy matrix C (Cij =1/(xi - yj))

# In[87]:


# ?


# #### 48. Print the minimum and maximum representable value for each numpy scalar type (★★☆)

# In[111]:


# fixed-size aliases
print("Integer-like types")
display(np.iinfo(np.int8), np.iinfo(np.uint8))  # 16, 32, 64
display(np.iinfo(np.intp), np.iinfo(np.uint8))

print("Floating point types")
display(np.finfo(np.float32), np.finfo(np.float_))  # 64

print("Complex types")
display(np.finfo(np.complex), np.finfo(np.complex_))  # 64, 128

# Platform-dependent definitions
# https://numpy.org/devdocs/user/basics.types.html


# #### 49. How to print all the values of an array? (★★☆)

# In[127]:


import sys

print(sys.maxsize)

np.set_printoptions(threshold=sys.maxsize)
Z = np.zeros((100,100))

print(Z)


# #### 50. How to find the closest value (to a given scalar) in a vector? (★★☆)

# In[142]:


v = np.random.uniform(0, 1, 100)
scalar = np.random.uniform(0, 1, 100)

closest_value_idx = (np.abs(scalar - v)).argmin()
closest_value = scalar[closest_value_idx]

display(closest_value)


# #### 51. Create a structured array representing a position (x,y) and a color (r,g,b) (★★☆)

# In[153]:


N = 5
a = np.ndarray(
    (N, N),
    [
        ('pos', [('x', np.float), ('y', np.float)]),
        ('color', [('r', np.float), ('g', np.float), ('b', np.float)])
    ]
)

display(a.dtype)


# #### 52. Consider a random vector with shape (100,2) representing coordinates, find point by point distances (★★☆)

# In[167]:


# ?


# #### 53. How to convert a float (32 bits) array into an integer (32 bits) in place?

# In[183]:


v = np.ndarray(shape=(5, 5), dtype=np.float32)

display(v.dtype)
display(v)

v = v.astype(np.int32, copy=False)

display(v.dtype)
display(v)


# #### 54. How to read the following file? (★★☆)

# ```
# 1, 2, 3, 4, 5
# 6,  ,  , 7, 8
#  ,  , 9,10,11
# ```

# In[ ]:





# #### 55. What is the equivalent of enumerate for numpy arrays? (★★☆)

# In[ ]:





# #### 56. Generate a generic 2D Gaussian-like array (★★☆)

# In[ ]:





# #### 57. How to randomly place p elements in a 2D array? (★★☆)

# In[ ]:





# #### 58. Subtract the mean of each row of a matrix (★★☆)

# In[ ]:





# #### 59. How to sort an array by the nth column? (★★☆)

# In[ ]:





# #### 60. How to tell if a given 2D array has null columns? (★★☆)

# In[ ]:





# #### 61. Find the nearest value from a given value in an array (★★☆)

# In[ ]:





# #### 62. Considering two arrays with shape (1,3) and (3,1), how to compute their sum using an iterator? (★★☆)

# In[ ]:





# #### 63. Create an array class that has a name attribute (★★☆)

# In[ ]:





# #### 64. Consider a given vector, how to add 1 to each element indexed by a second vector (be careful with repeated indices)? (★★★)

# In[ ]:





# #### 65. How to accumulate elements of a vector (X) to an array (F) based on an index list (I)? (★★★)

# In[ ]:





# #### 66. Considering a (w,h,3) image of (dtype=ubyte), compute the number of unique colors (★★★)

# In[ ]:





# #### 67. Considering a four dimensions array, how to get sum over the last two axis at once? (★★★)

# In[ ]:





# #### 68. Considering a one-dimensional vector D, how to compute means of subsets of D using a vector S of same size describing subset  indices? (★★★)

# In[ ]:





# #### 69. How to get the diagonal of a dot product? (★★★)

# In[ ]:





# #### 70. Consider the vector \[1, 2, 3, 4, 5\], how to build a new vector with 3 consecutive zeros interleaved between each value? (★★★)

# In[ ]:





# #### 71. Consider an array of dimension (5,5,3), how to mulitply it by an array with dimensions (5,5)? (★★★)

# In[ ]:





# #### 72. How to swap two rows of an array? (★★★)

# In[ ]:





# #### 73. Consider a set of 10 triplets describing 10 triangles (with shared vertices), find the set of unique line segments composing all the  triangles (★★★)

# In[ ]:





# #### 74. Given an array C that is a bincount, how to produce an array A such that np.bincount(A) == C? (★★★)

# In[ ]:





# #### 75. How to compute averages using a sliding window over an array? (★★★)

# In[ ]:





# #### 76. Consider a one-dimensional array Z, build a two-dimensional array whose first row is (Z\[0\],Z\[1\],Z\[2\]) and each subsequent row is  shifted by 1 (last row should be (Z\[-3\],Z\[-2\],Z\[-1\]) (★★★)

# In[ ]:





# #### 77. How to negate a boolean, or to change the sign of a float inplace? (★★★)

# In[ ]:





# #### 78. Consider 2 sets of points P0,P1 describing lines (2d) and a point p, how to compute distance from p to each line i  (P0\[i\],P1\[i\])? (★★★)

# In[ ]:





# #### 79. Consider 2 sets of points P0,P1 describing lines (2d) and a set of points P, how to compute distance from each point j (P\[j\]) to each line i (P0\[i\],P1\[i\])? (★★★)

# In[ ]:





# #### 80. Consider an arbitrary array, write a function that extract a subpart with a fixed shape and centered on a given element (pad with a `fill` value when necessary) (★★★)

# In[ ]:





# #### 81. Consider an array Z = \[1,2,3,4,5,6,7,8,9,10,11,12,13,14\], how to generate an array R = \[\[1,2,3,4\], \[2,3,4,5\], \[3,4,5,6\], ..., \[11,12,13,14\]\]? (★★★)

# In[ ]:





# #### 82. Compute a matrix rank (★★★)

# In[ ]:





# #### 83. How to find the most frequent value in an array?

# In[ ]:





# #### 84. Extract all the contiguous 3x3 blocks from a random 10x10 matrix (★★★)

# In[ ]:





# #### 85. Create a 2D array subclass such that Z\[i,j\] == Z\[j,i\] (★★★)

# In[ ]:





# #### 86. Consider a set of p matrices wich shape (n,n) and a set of p vectors with shape (n,1). How to compute the sum of of the p matrix products at once? (result has shape (n,1)) (★★★)

# In[ ]:





# #### 87. Consider a 16x16 array, how to get the block-sum (block size is 4x4)? (★★★)

# In[ ]:





# #### 88. How to implement the Game of Life using numpy arrays? (★★★)

# In[ ]:





# #### 89. How to get the n largest values of an array (★★★)

# In[ ]:





# #### 90. Given an arbitrary number of vectors, build the cartesian product (every combinations of every item) (★★★)

# In[ ]:





# #### 91. How to create a record array from a regular array? (★★★)

# In[ ]:





# #### 92. Consider a large vector Z, compute Z to the power of 3 using 3 different methods (★★★)

# In[ ]:





# #### 93. Consider two arrays A and B of shape (8,3) and (2,2). How to find rows of A that contain elements of each row of B regardless of the order of the elements in B? (★★★)

# In[ ]:





# #### 94. Considering a 10x3 matrix, extract rows with unequal values (e.g. \[2,2,3\]) (★★★)

# In[ ]:





# #### 95. Convert a vector of ints into a matrix binary representation (★★★)

# In[ ]:





# #### 96. Given a two dimensional array, how to extract unique rows? (★★★)

# In[ ]:





# #### 97. Considering 2 vectors A & B, write the einsum equivalent of inner, outer, sum, and mul function (★★★)

# In[ ]:





# #### 98. Considering a path described by two vectors (X,Y), how to sample it using equidistant samples (★★★)?

# In[ ]:





# #### 99. Given an integer n and a 2D array X, select from X the rows which can be interpreted as draws from a multinomial distribution with n degrees, i.e., the rows which only contain integers and which sum to n. (★★★)

# In[ ]:





# #### 100. Compute bootstrapped 95% confidence intervals for the mean of a 1D array X (i.e., resample the elements of an array with replacement N times, compute the mean of each sample, and then compute percentiles over the means). (★★★)

# In[ ]:





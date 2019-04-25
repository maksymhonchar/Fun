import numpy as np

# The core functionality of NumPy is the ndarray class, a multidimensional (n-dimensional) array.
# All elements of the array must be of the same type.
x = np.array([[1, 2, 3], [4, 5, 6]])
print("x:\n{}".format(x))

from scipy import sparse

# The most important part of SciPy is scipy.sparse: this provides sparse matrices, which are another representation that is used for data in scikit-learn.
# Sparse matrices are used whenever we want to store a 2D array that contains mostly zeros.

eye = np.eye(4)
print("NumPy array:\n{}".format(eye))

# Convert the NumPy array to a SciPy sparse matrix in CSR format.
# Only the nonzero entries are stored.
sparse_matrix = sparse.csr_matrix(eye)
print("SciPy sparse CSR matrix:\n{}".format(sparse_matrix))

# Usually it is not possible to create dense representations of sparse data as they would not fit into memory, so we need to create sparse representations directl.y
# Create sparse matrix using COO representation format:
data = np.ones(5)
row_indices = np.arange(5)
col_indices = np.arange(5)
eye_coo = sparse.coo_matrix(
    (data, (row_indices, col_indices))
)
print("COO representation:\n{0}\nNumpy representation:\n{1}".format(eye_coo, data))

# matplotlib is the primary scientific library in Python. It provides funcs for making publication-qualilty visualizations such as line charts, histograms, scatter plots, and so on.
# When working in JupyterNotebook, you can show figures by using the [%matplotlib notebook] and [%matplotlib inline] commands.
import matplotlib.pyplot as plt
x = np.linspace(-10, 10, 1000)  # sequence of numbers [-10;10] with 100 steps in between
y = np.sin(x)
plt.plot(x, y, marker='x')
plt.show()

# pandas is a lib for data wrangling and analysis.
# pandas DataFrame is a table, similar to an Excel spreadsheet.
# In contrast to NumPy, which requires that all entries in an array be of the same type, pandas allow each column to have a separate type.
# pandas also provides ability to ingest from a great variety of file formats/databases: sql, excel files, csv files.
import pandas as pd
data = {
    "Name": ["max", "alex", "mikhail", "kostya"],
    "Location": ["NY", "Paris", "Berlin", "Kyiv"],
    "Age": [10, 20, 30, 40]
}
data_pandas = pd.DataFrame(data)
print(data_pandas)
print(data_pandas.to_string())  # same as without to_string()
print(data_pandas[data_pandas.Age > 25])

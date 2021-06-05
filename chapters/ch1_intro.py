# NumPy
import numpy as np

x = np.array([[1, 2, 3], [4, 5, 6]])
print('x:\n{}'.format(x))

# SciPy
from scipy import sparse

# Create a 2D Numpy array with a diagonal of ines, and zeros everywhere else
eye = np.eye(4)
eye

# Convert the Numpy array to a SciPy sparse matrix
# in compressed sparse row (CSR) format
# Only the nonzero entries are stored
sparse_matrix = sparse.csr_matrix(eye)
sparse_matrix
print('\nSciPy sparse CSR matrix:\n{}'.format(sparse_matrix))

data = np.ones(4)
data

row_indices = np.arange(4)
col_indices = np.arange(4)
eye_coo = sparse.coo_matrix((data, (row_indices, col_indices)))
eye_coo
print('COO representation:\n{}'.format(eye_coo))

# matplotlib
# %matplotlib notebook
%matplotlib inline
import matplotlib.pyplot as plt

# Genereate a sequence of numbers from -10 to 10 with 100 steps in between
x = np.linspace(-10, 10, 100)
# Create a second array using sine
y = np.sin(x)
# The plot function makes a line chart of one array against another
plt.plot(x, y, marker='x')


# pandas
import pandas as pd

# Create a simple dataset of people
data = {'Name': ['John', 'Anna', 'Peter', "Linda"],
        'Location': ['New York', 'Paris', 'Berlin', 'London'],
        'Age': [24, 13, 53, 33]}

data_pandas = pd.DataFrame(data)
# IPython.display allows "pretty printing" of data frames
# in the Jupyter notebook
display(data_pandas)
data_pandas

data_pandas[data_pandas.Age > 30]

# mglearn
import mglearn

# Versions Used in this Book
import sys # python version
sys.version

import pandas as pd
pd.__version__

import matplotlib
matplotlib.__version__

import numpy as np
np.__version__

import scipy as sp
sp.__version__

import IPython
IPython.__version__

import sklearn
sklearn.__version__

# A First Application: Classifying Iris Species
# Meet the Data
from sklearn.datasets import load_iris
iris_dataset = load_iris()
type(iris_dataset)
# help(sklearn.utils.Bunch)
iris_dataset.keys()
print(iris_dataset['DESCR'][:200])

iris_dataset['feature_names']
iris_dataset['data'][:5]
iris_dataset['data'].shape
iris_dataset['target_names']
iris_dataset['target']

# Measuring Success: Training and Testing Data
from sklearn.model_selection import train_test_split
X = iris_dataset['data']
y = iris_dataset['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75,
                                                    random_state=0)
# print(train_test_split.__doc__)
X_train.shape
y_train.shape
X_test.shape
y_test.shape

iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
iris_dataframe
# Create a scatter matrix from the dataframe, color by y_train
grr = pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(10, 10),
                                 marker='o', hist_kwds={'bins': 20}, s=60,
                                 alpha=.8, cmap=mglearn.cm3)


# Building Your First Model: k-Nearest Neighbors
from  sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

# Making Predictions
X_new = np.array([[5, 2.9, 1, 0.2]])
X_new.shape

prediction = knn.predict(X_new)
prediction
iris_dataset.target_names[prediction]
iris_dataset['target_names'][prediction]

# Evaluating the Model
y_pred = knn.predict(X_test)
y_pred
np.mean(y_pred == y_test)
knn.score(X_test, y_test)

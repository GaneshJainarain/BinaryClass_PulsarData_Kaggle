## Playground Series - Season 3, Episode 10
### `Binary Classification with a Pulsar Dataset`

The dataset for this competition (both train and test) was generated from a deep learning model trained on the Pulsar Classification. 

#### `Files`

- train.csv - the training dataset; Class is the (binary) target
- test.csv - the test dataset; your objective is to predict the probability of Class
(whether the observation is a pulsar)
- sample_submission.csv - a sample submission file in the correct format

### `Data Science Framework & Process`

- Define the Problem
- Gather the Data
- Prepare Data for Consumption
- Perform Exploratory Analysis
- Model Data
- Validate and Implement Data Model
- Optimize and Strategize

### `Defining Our Problem`

`Predict the probability of the variable Class`

Pulsars are rapidly spinning neutron stars, extremely dense stars composed almost entirely of neutrons and having a diameter of only 20 km (12 miles) or less. Pulsar masses range between 1.18 and 1.97 times that of the Sun, but most pulsars have a mass 1.35 times that of the Sun.

Pulsars are a rare type of Neutron star that produce radio emission detectable here on Earth. They are of considerable scientific interest as probes of space-time, the inter-stellar medium, and states of matter . Neutron stars are very dense, and have short, regular rotational periods. This produces a very precise interval between pulses that ranges from milliseconds to seconds for an individual pulsar. Pulsars are believed to be one of the candidates for the source of ultra-high-energy cosmic rays.

### `Gather the Data`

The data is given to us via Kaggle, Download at:
(Kaggle Pulsar Dataset)[https://www.kaggle.com/competitions/playground-series-s3e10/data]

### `Prepare Data for Consumption`

#### `Import Libraries`

```python

#load packages
import sys #access to system parameters https://docs.python.org/3/library/sys.html
print("Python version: {}". format(sys.version))

import pandas as pd #collection of functions for data processing and analysis modeled after R dataframes with SQL like features
print("pandas version: {}". format(pd.__version__))

import matplotlib #collection of functions for scientific and publication-ready visualization
print("matplotlib version: {}". format(matplotlib.__version__))

import numpy as np #foundational package for scientific computing
print("NumPy version: {}". format(np.__version__))

import scipy as sp #collection of functions for scientific computing and advance mathematics
print("SciPy version: {}". format(sp.__version__)) 

import IPython
from IPython import display #pretty printing of dataframes in Jupyter notebook
print("IPython version: {}". format(IPython.__version__)) 

import sklearn #collection of machine learning algorithms
print("scikit-learn version: {}". format(sklearn.__version__))

#misc libraries
import random
import time

#ignore warnings
import warnings
warnings.filterwarnings('ignore')
print('-'*25)

```

```
Python version: 3.8.3 (v3.8.3:6f8c8320e9, May 13 2020, 16:29:34) 
[Clang 6.0 (clang-600.0.57)]
pandas version: 1.5.3
matplotlib version: 3.7.1
NumPy version: 1.24.2
SciPy version: 1.10.1
IPython version: 8.11.0
scikit-learn version: 1.2.1
-------------------------
```

#### `Load Data Modelling Libraries`
We will use the popular scikit-learn library to develop our `machine learning algorithms`. In sklearn, algorithms are called Estimators and implemented in their own classes. For `data visualization`, we will use the matplotlib and seaborn library. Below are common classes to load.

```python

#Common Model Algorithms
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier

#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

#Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.tools.plotting import scatter_matrix

#Configure Visualization Defaults
#%matplotlib inline = show plots in Jupyter Notebook browser
%matplotlib inline
mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize'] = 12,8

```
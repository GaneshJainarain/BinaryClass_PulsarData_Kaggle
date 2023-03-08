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

#### `Meet and Greet Data`

This is the meet and greet step. Get to know your data by first name and learn a little bit about it. What does it look like (datatype and values), what makes it tick (independent/feature variables(s)), what's its goals in life (dependent/target variable(s)). Think of it like a first date, 

#### `Our Data`

`17898 entries`

Data can be useful for prediction models of classification.

`COLUMNS:`
Based on Integrated Profile of Observation

- Mean_Integrated: Mean of Observations

- SD: Standard deviation of Observations

- EK: Excess kurtosis of Observations

- Skewness: In probability theory and statistics, skewness is a measure of the asymmetry of the probability distribution of a real-valued random variable about its mean. Skewness of Observations.

- Mean _ DMSNR _ Curve: Mean of DM SNR CURVE of Observations

- SD _ DMSNR _ Curve: Standard deviation of DM SNR CURVE of Observations

- EK _ DMSNR _ Curve: Excess kurtosis of DM SNR CURVE of Observations

- Skewness _ DMSNR _ Curve: Skewness of DM SNR CURVE of Observations

- Class: Class 0 - 1

`WHAT IS DM SNR CURVE:`

Radio waves emitted from pulsars reach earth after traveling long distances in space which is filled with free electrons. 
The important point is that pulsars emit a wide range of frequencies, and the amount by which the electrons slow down the wave depends on the frequency. 
Waves with higher frequency are sowed down less as compared to waves with higher frequency. It means dispersion.

`TARGET:`

`Class`
   - 0 -- It is not
   - 1 -- It is

#### `Train/Test`

```python

data_raw = pd.read_csv('../input/train.csv')

#a dataset should be broken into 3 splits: train, test, and (final) validation
#the test file provided is the validation file for competition submission
#we will split the train set into train and test data in future sections
data_val  = pd.read_csv('../input/test.csv')

```

```python

#to play with our data we'll create a copy
#remember python assignment or equal passes by reference vs values, so we use the copy function: https://stackoverflow.com/questions/46327494/python-pandas-dataframe-copydeep-false-vs-copydeep-true-vs

data1 = data_raw.copy(deep = True)

#however passing by reference is convenient, because we can clean both datasets at once
data_cleaner = [data1, data_val]


#preview data

print (data_raw.info()) #https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.info.html
#data_raw.head() #https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.head.html
#data_raw.tail() #https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.tail.html

print(data_raw.sample(10)) 

#https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sample.html

```
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 117564 entries, 0 to 117563
Data columns (total 10 columns):
 #   Column                Non-Null Count   Dtype  
---  ------                --------------   -----  
 0   id                    117564 non-null  int64  
 1   Mean_Integrated       117564 non-null  float64
 2   SD                    117564 non-null  float64
 3   EK                    117564 non-null  float64
 4   Skewness              117564 non-null  float64
 5   Mean_DMSNR_Curve      117564 non-null  float64
 6   SD_DMSNR_Curve        117564 non-null  float64
 7   EK_DMSNR_Curve        117564 non-null  float64
 8   Skewness_DMSNR_Curve  117564 non-null  float64
 9   Class                 117564 non-null  int64  
dtypes: float64(8), int64(2)
memory usage: 9.0 MB
None


```
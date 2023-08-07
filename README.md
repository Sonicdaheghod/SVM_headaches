# SVM_headaches
by Megan Tran

Download the necessary packages:
```
pip install scikit-learn
pip install pandas
pip install numpy
pip install matplotlib
```
Check to see if version of Python/Python3 (if on jupyter Notebook) is used, this is to ensure packages work properly. Python 3.8 or better is recommended.

```
import sys
sys.version
```

Import the following packages and libraries:

```
# Data processing
import pandas as pd
import numpy as np

# Visualizing Dataset
import matplotlib as plt
import matplotlib.pyplot as plt

# Model
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import SVC
```
## Using the Program
1) Data Cleaning and Preprocessing

In dataset, the values under "Type" were strings, so a dictionary was created to change each type of migrane to numerical values.

```
#Create dictionary to turn labels to values

migrane_labels_to_numbers = {
    
    "Typical aura with migraine" : 1,
    "Migraine without aura" : 2,
    "Basilar-type aura" : 3,
    "Sporadic hemiplegic migraine": 4,
    "Familial hemiplegic migraine" : 5,
    "Typical aura without migraine" : 6,
    "Other" : 7
    
}

#relabel each list of labels to values
pd_migrane["Type"] = pd_migrane["Type"].map(migrane_labels_to_numbers)

#make updated csv
pd_migrane.to_csv("migrane_dataMod.csv", index = False)

#open new csv

pd_mod_migrane = pd.read_csv("migrane_dataMod.csv").head()

pd_mod_migrane

```

2) SVM Model

Training and testing data points from dataset need to be made. I wanted the SVM model to use data under the columns below to predict the type of migrane a patient has.
```

Var_features = pd_mod_migrane[['Age', 'Duration', 'Frequency', 'Location', 'Character', 'Intensity',
       'Nausea', 'Vomit', 'Phonophobia', 'Photophobia', 'Visual', 'Sensory',
       'Dysphasia', 'Dysarthria', 'Vertigo', 'Tinnitus', 'Hypoacusis',
       'Diplopia', 'Defect', 'Ataxia', 'Conscience', 'Paresthesia', 'DPF']
    
]

```

Here the independent and dependent variables are made.
```
#List Independent Var
IV_migrane = np.asarray(Var_features)

#List Dependent Var
DV_migrane = np.asarray(pd_mod_migrane["Type"])
```
Now we create the train/test split. Smaller test size means the model has more to train with, which means it can make more accurate predictions than training with fewer datapoints.
```
from sklearn.model_selection import train_test_split

IV_migrane_train, IV_migrane_test, DV_migrane_train, DV_migrane_test=  train_test_split(IV_migrane,DV_migrane, test_size = .25, random_state = 4)
```
3) Evaluating Model

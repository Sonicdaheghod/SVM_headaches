# SVM_headaches
by Megan Tran

## Table of Contents
* [Purpose of Program](#Purpose-of-program)
* [Technologies](#technologies)
* [Setup](#setup)
* [Using the Program](#Using-the-Program)
* [Credits](#Credits)

## Purpose of Program

* Practicing SVM modelling using migrane dataset.

## Technologies
Languages/ Technologies used:

* Jupyter Notebook

* Python3

## Setup

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

In dataset, the values under "Type" were strings, so a dictionary was created to change each type of migrane to numerical values. Ot was then saved as a modified version of the original dataset.

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

```
![image](https://github.com/Sonicdaheghod/SVM_headaches/assets/68253811/f759ac63-a75b-4db7-a014-1b69a0d5978d)

2) SVM Model

* Training and testing data points from dataset need to be made. I wanted the SVM model to use data under the columns below to predict the type of migrane a patient has.
```

Var_features = pd_mod_migrane[['Age', 'Duration', 'Frequency', 'Location', 'Character', 'Intensity',
       'Nausea', 'Vomit', 'Phonophobia', 'Photophobia', 'Visual', 'Sensory',
       'Dysphasia', 'Dysarthria', 'Vertigo', 'Tinnitus', 'Hypoacusis',
       'Diplopia', 'Defect', 'Ataxia', 'Conscience', 'Paresthesia', 'DPF']
    
]

```

* Independent and Dependent variables made for train_test_split. This allows the model to predict the type of migrane a patient has based on various features about a patient as seen in Var_features.
* Now we create the train/test split. Smaller test size means the model has more to train with, which means it can make more accurate predictions than training with fewer datapoints.
```
from sklearn.model_selection import train_test_split

IV_migrane_train, IV_migrane_test, DV_migrane_train, DV_migrane_test=  train_test_split(IV_migrane,DV_migrane, test_size = .25, random_state = 4)
```

* Parameters for SVM model were

```
(kernel = "rbf", gamma = 'auto', C=1)
```

I used rbf so the classification would be better with my nonlinear dataset.

3) Evaluating Model

* Classification report was executed for my SVM model
<img width="420" alt="image" src="https://github.com/Sonicdaheghod/SVM_headaches/assets/68253811/f738f22e-9e6b-4075-abcd-ae2e67380d34">

** In the future, the zero_division code should be implemented to fix the zeros in the Type values 3,4,5, and 7. This occured because the model did not classify the test data into those categories. Thus, the zero led to a undefined answer when plugged into the formula to determine their precision, f1- socre, and recall.

## Credits
Source code tutorial by [Cloud and ML Online](https://youtu.be/7sz4WpkUIIs?t=1819)
Dataset by [EMIR YARKIN YAMAN](https://www.kaggle.com/datasets/weinoose/migraine-classification)


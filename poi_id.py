#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

import pandas as pd
import numpy as np
import math
from itertools import combinations

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Storing data in a dataframe to simplify data manipulation
enron_df = pd.DataFrame.from_dict(data_dict, orient='index')

### ------------------------------------ ###

### Replacing NaN's
enron_df.replace(to_replace={'NaN':0}, inplace=True)

### Removing identified outliers
enron_df.drop('TOTAL', inplace=True)
enron_df.drop('LOCKHART EUGENE E', inplace=True)
enron_df.drop('THE TRAVEL AGENCY IN THE PARK', inplace=True)

### Removing 'poi' and the categorical variable 'email_address'
cols_to_analyse = [x for x in enron_df.columns if x not in ['poi', 'email_address']]
### Changing the outliers (values higher than the mean plus two times the standard deviation)
for col in cols_to_analyse:
    threshold = 2*enron_df[col].std()+enron_df[col].mean()
    ### Setting outliers to 0
    enron_df.loc[enron_df[col] > threshold, col] = 0

enron_df['from_poi_percent'] = enron_df['from_poi_to_this_person'] / enron_df['to_messages']
enron_df['to_poi_percent'] = enron_df['from_this_person_to_poi'] / enron_df['from_messages']
enron_df['more_to_than_from'] = np.where(enron_df['to_messages'] / enron_df['from_messages'] > 1, 1, 0)

### Replacing NaN's
for col in ['from_poi_percent','to_poi_percent','more_to_than_from']:
    enron_df.loc[pd.isnull(enron_df[col]), col] = 0
    enron_df.loc[~np.isfinite(enron_df[col]), col] = 0

### Splitting data into Train x Test
from sklearn import model_selection

enron_df.drop(['email_address'], axis=1, inplace=True)

features = enron_df.drop(['poi'], axis=1)
labels = enron_df['poi']

features_train, features_test, labels_train, labels_test = model_selection.train_test_split(
    features.values, labels, test_size=0.3, random_state=42)

### Dealing with negative values for further use
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(features_train)

features_train = scaler.transform(features_train)
features_test = scaler.transform(features_test)

### ------------------------------------ ###

## Selecting which features are going to be used
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

### Number of features to be selected
k = 7

### Creates and fits selector
selector = SelectKBest(chi2, k=k)
selector.fit(features_train, labels_train)

### Get idxs of columns to keep
selected_idx = selector.get_support(indices=True)

### Applies selection over features
features_train_selected = selector.transform(features_train)
features_test_selected = selector.transform(features_test)

columns_list = enron_df[selected_idx].columns.tolist()

### ------------------------------------ ###

from sklearn.model_selection import GridSearchCV

### Testing different Machine Learning approaches
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

parameters = {
    'learning_rate': [1, 1.2, 1.25, 1.3, 1.35, 1.4],
    'n_estimators': [5],
    'random_state': [511]    
}

ada = AdaBoostClassifier()
### Using GridSearchVC to validate the classifier
clf = GridSearchCV(ada, parameters, scoring=['precision', 'recall'], refit='precision')

clf.fit(features_train_selected, labels_train)

### The scores can be seen by uncommenting the next line
# test_classifier(clf, my_dataset, features_list)

### ------------------------------------ ###

### Testing classifier with new features
my_dataset = enron_df.to_dict(orient='index')

### Defining a list of features that contains the new variables
### This new list has the same length used by the SelectKBest
features = enron_df[['from_poi_percent', 'to_poi_percent', 'more_to_than_from',\
                     'long_term_incentive', 'expenses', 'shared_receipt_with_poi']]
labels = enron_df['poi']

### Splitting and Scaling the data in the same way as before
features_train, features_test, labels_train, labels_test = model_selection.train_test_split(
    features.values, labels, test_size=0.3, random_state=42)

scaler = MinMaxScaler()
scaler.fit(features_train)

features_train = scaler.transform(features_train)
features_test = scaler.transform(features_test)

### Testing classifier
clf = AdaBoostClassifier(learning_rate=1.2, n_estimators=5, random_state=511)
clf.fit(features_train, labels_train)

features_list = ['poi'] + features.columns.tolist()

### The scores can be seen by uncommenting the next line
# test_classifier(clf, my_dataset, features_list)

dump_classifier_and_data(clf, my_dataset, features_list)
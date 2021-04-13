# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# To work with Dataframe
import pandas as pd

# To perform numerical operation
import numpy as np

# To visualize data
import seaborn as sns

# To partition the data
from sklearn.model_selection import train_test_split

# Importing library for logistic regression
from sklearn.linear_model import LogisticRegression

# Importing performance metrics - accuracyscore & confusion
from sklearn.metrics import accuracy_score,confusion_matrix

# Importing data
data_income = pd.read_csv('income.csv')

# Creating a copy of original data
data = data_income.copy()

# To check the datatypes of each variables
print(data.info())

# Check for missing values
data.isnull()

# Getting number of missing values across each variale
print(data.isnull().sum())

# Getting ststistical description of data
summary_num=data.describe()
print(summary_num)

# Getting summary of categrical data
summary_cate=data.describe(include = "O")
print(summary_cate)

# Getting frequecy of each variable
data['JobType'].value_counts()
data['occupation'].value_counts()

# Checking for unique classes to know the \
# exact representation of missing values
print(np.unique(data['JobType']))
print(np.unique(data['occupation']))

# Again read data by including 'na_values' for missimng values
data=pd.read_csv('income.csv', na_values=[" ?"])

# Data preprocessing
# To find missing values(nan) across each variable
data.isnull().sum()

# Inspect missing data(subsetting the data i.e. create subset of original dataframe as missing)
missing = data[data.isnull().any(axis=1)]
# axis=1 => to consider at least one column value is missing

# Deleting all rows having missing values
data2=data.dropna(axis=0)
# axis=0 => represents missing values in a row

# Relation(correlation) between numerical variable i.e. independent variable
correlation = data2.corr()

# Cross table(relationship between categorical variable) and data visalization

# Extracting the column names
data2.columns

# Getting gender proportiom table
gender = pd.crosstab(index = data2['gender'], columns ='count', normalize = True)
print(gender)

# Getting gender vs salary status 
gender_salstat = pd.crosstab(index = data2['gender'], columns = data2['SalStat'] , margins = True,normalize = 'index')
print(gender_salstat)

# Frequency distribution of output variable i.e. salary status
SalStat = sns.countplot(data2['SalStat'])
# According to ferequency distribution 75% peoples salary status is <=5000 \
# and 25% peoples salary status is >50000

# Plot histogram to get the frequency distribution age
sns.distplot(data2['age'],bins=10,kde=True)
# People with age between 20-45 are high in frequency

# Bivariate analysis between age and salstat
sns.boxplot('SalStat', 'age', data=data2)
data2.groupby('SalStat')['age'].median()
# People with age 25-35 more likely to earn <=50000
# People with age 35-50 more likely to earn >50000

#=======================================================
# Logistic Regression
#=======================================================

# Rendering the salary status to 0 and 1
# we are assingin here 0 for <=50000 and 1 for >50000
data2['SalStat'] = data2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2)

# One hot coding (convert categorical variable into dummy variable)
new_data = pd.get_dummies(data2, drop_first=True)

# Storing column names of new_data ino list
columns_list = list(new_data.columns)
print(columns_list)

# Separating input from data
# Excluding SalStat from data
features = list(set(columns_list)-set(['SalStat']))
print(features)

# Storing the outputvalue in y(corresponding numerical values)
y = new_data['SalStat'].values
print(y)

# Storig the values from input features
x = new_data[features].values
print(x)

# Splitting the data into train ant test
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3,random_state=0)

# Make an instance of model
logistic = LogisticRegression()

# Fitting values for x and y i.e. we fit the model on train data set
logistic.fit(train_x, train_y)
logistic.coef_

# Prediction from test data
prediction = logistic.predict(test_x)
print(prediction)

# Confusion Matrix
confusion_matrix = confusion_matrix(test_y,prediction)
print(confusion_matrix)

# Calculating accuracy score for confusion matrix(model)
accuracy_score = accuracy_score(test_y,prediction)
print(accuracy_score)

# Printing missclassified values
print('Missclassified samples %d'% (test_y != prediction).sum())

#=======================================================================
# KNN
#=====================================================================

# importing library of KNN
from sklearn.neighbors import KNeighborsClassifier

# Importing library for plotting
import matplotlib.pyplot as plt

# Instance of KNN classifier
KNN_classifier = KNeighborsClassifier(n_neighbors=5)
# Above KNN classfier consider five nrighbors to classify data

# Fitting values for x and y
KNN_classifier.fit(train_x,train_y)

# Predicting test value for model
Prediction = KNN_classifier.predict(test_x)

# Performance metrics check
confusion_matrix = confusion_matrix(test_y,prediction)
print(confusion_matrix)

# Calculating accuracy
accuracy_score = accuracy_score(test_y,prediction)
print(accuracy_score)

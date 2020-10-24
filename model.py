# Importing essential libraries
import numpy as np
import pandas as pd
import pickle

# Loading the dataset
df = pd.read_csv('D:\Data science\Assignments docs\Random Forest\Company_data.csv')

# Converting the salary into categories of 'high' and 'low
sales = pd.cut(df.Sales, bins=[0,10,20], labels=["low","high"])

# Combining the converted values into the original dataset and removing earlier column
df.insert(1,'sales',sales)
df = df.drop(columns =["Sales"],axis=1)
df.head()

# Obtaining the dummy variables for 'ShelveLoc','Urban','US'
df = pd.get_dummies(df, columns=['ShelveLoc','Urban','US'], drop_first = True)
df.head()

# Replacing NA value in 'sales'
df['sales'].fillna('high', inplace=True)

# Model Building
from sklearn.model_selection import train_test_split
X = df.drop(columns='sales')
y = df['sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Creating Random Forest Model
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100)
classifier.fit(X_train, y_train)

# Creating a pickle file for the classifier
filename = 'Sales-prediction-rfc-model.pkl'
pickle.dump(classifier, open(filename, 'wb'))

#Imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler

#Read the dataset
dataset = pd.read_csv(r'data/winequality-red.csv')
#print(dataset.head(5))

#NA values in the dataset
#print("Count of NA values:")
#print(dataset.isna().sum())

print(set(dataset['quality']))

dataset['quality'] = dataset['quality'].map({
        3 : 0,
        4 : 0,
        5 : 1,
        6 : 1,
        7 : 2,
        8 : 2
        })

X = dataset.iloc[:, :11].values
y = dataset.iloc[:, [11]].values

#Split data into training and testing split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0, stratify = y)

#Preprocess the dataset
scaler = StandardScaler()
#scaler = Normalizer()
#scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Train a classifier
classifier = GaussianNB()
classifier.fit(X_train, y_train.ravel())

#Predcitions
y_pred = classifier.predict(X_test)

#Results
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:")
print(cm)
print("Accuracy:", accuracy_score(y_test, y_pred))
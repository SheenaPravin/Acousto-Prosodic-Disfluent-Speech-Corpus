# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 11:15:26 2018

@author: HP
"""


import pandas as pd
dataset = pd.read_csv('Acousto-Prosody_Final.csv')  
dataset.head()  
X = dataset.iloc[1:760, 0:42].values  
y = dataset.iloc[1:760, 43].values  
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Fit only to the training data
scaler.fit(X_train)
StandardScaler(copy=True, with_mean=True, with_std=True)
# Now apply the transformations to the data:
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
#svm = LinearSVC(random_state=0,  C=1)
svm = SVC(kernel='poly', random_state=0, gamma=0.1, C=1)
svm.fit(X_train, y_train)
#print (svm.fit.n_support_)
predictions = svm.predict(X_test)
print('The accuracy of the SVM classifier on training data is {:.2f}'.format(svm.score(X_train, y_train)))
print('The accuracy of the SVM classifier on test data is {:.2f}'.format(svm.score(X_test, y_test)))
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

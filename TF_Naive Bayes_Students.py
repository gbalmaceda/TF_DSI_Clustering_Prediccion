# -*- coding: utf-8 -*-
"""
Created on Wed Jun 06 16:55:58 2018

@author: Gustavo Balmaceda
"""
import pandas as pd 
import matplotlib.pyplot as plt 

algebra_05_06__Stdnts_train = pd.read_csv('algebra_2005_2006_Student_train.csv',sep=';')
algebra_05_06__Stdnts_train.head()

# preprocessing.
#X = algebra_05_06__Stdnts_train[['Cuenta de Problem Name','Cuenta de Step Name','Suma de Step Duration (sec)','Suma de Correct Step Duration (sec)','Suma de Error Step Duration (sec)','Suma de Problem View','Suma de Corrects','Suma de Incorrects','Suma de Hints','Duracion Promedia por paso (sec)','Durancion promedia pasos correctos','Duracion promedia de pasos incorrectos','Promedio de pasos resueltos en primer intento']]
X = algebra_05_06__Stdnts_train[['Cuenta de Step Name','Suma de Step Duration (sec)','Suma de Correct Step Duration (sec)','Suma de Corrects','Suma de Incorrects','Suma de Hints','Duracion Promedia por paso (sec)','Durancion promedia pasos correctos','Duracion promedia de pasos incorrectos','Promedio de pasos resueltos en primer intento']]


##########Pruebas###########
#X = algebra_05_06__Stdnts_train[['Cuenta de Step Name','Suma de Step Duration (sec)','Suma de Correct Step Duration (sec)','Suma de Error Step Duration (sec)','Suma de Problem View','Suma de Corrects','Suma de Incorrects','Suma de Hints']]
#X = algebra_05_06__Stdnts_train[['Suma de Step Duration (sec)','Suma de Correct Step Duration (sec)','Suma de Corrects','Duracion Promedia por paso (sec)','Durancion promedia pasos correctos','Promedio de pasos resueltos en primer intento']]
#X = algebra_05_06__Stdnts_train[['Cuenta de Step Name','Suma de Step Duration (sec)','Suma de Correct Step Duration (sec)','Suma de Corrects','Durancion promedia pasos correctos','Duracion promedia de pasos incorrectos','Promedio de pasos resueltos en primer intento']]


# standarization
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(X) 
X_scaled = scaler.transform(X) 
# round
y = algebra_05_06__Stdnts_train['Suma de Correct First Attempt'] 
y_round = [ round(e, 0) for e in y ]

# sample a training set while holding out 40% of the data for testing (evaluating) our classifier:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_round, test_size=0.4)

#Create a Gaussian Classifier
from sklearn.naive_bayes import GaussianNB 

# Parametrization
model =GaussianNB()
# training the model, i.e., likelihood computing 
model.fit(X_train,y_train)
# prediction with the same data
y_pred = model.predict(X_test)

# metrics calculation 
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test,y_pred) 
print "Error Measure ", mae

# PLOTTING 
import numpy as np 
xx = np.stack(i for i in range(len(y_test)))
plt.scatter(xx, y_test, c='r', label='data') 
plt.plot(xx, y_pred, c='g', label='prediction')
plt.axis('tight') 
plt.legend() 
plt.title("Gaussian NaiveBayes") 
plt.show()
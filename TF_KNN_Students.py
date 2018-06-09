# -*- coding: utf-8 -*-
"""
Created on Wed Jun 06 16:55:58 2018

@author: Gustavo Balmaceda
"""
import pandas as pd 
import matplotlib.pyplot as plt 

# Lectura y carga de datos de entrenamiento
algebra_05_06__Stdnts_train = pd.read_csv('algebra_2005_2006_Student_train.csv',sep=';')
algebra_05_06__Stdnts_train.head()

# Definición de características a considerar
# Características: 'Cuenta de Step Name','Suma de Step Duration (sec)','Suma de Correct Step Duration (sec)','Suma de Corrects','Suma de Incorrects','Suma de Hints','Duracion Promedia por paso (sec)','Durancion promedia pasos correctos','Duracion promedia de pasos incorrectos','Promedio de pasos resueltos en primer intento'

X = algebra_05_06__Stdnts_train[['Cuenta de Step Name','Suma de Step Duration (sec)','Suma de Correct Step Duration (sec)','Suma de Corrects','Suma de Incorrects','Suma de Hints','Duracion Promedia por paso (sec)','Durancion promedia pasos correctos','Duracion promedia de pasos incorrectos','Promedio de pasos resueltos en primer intento']]
y = algebra_05_06__Stdnts_train['Suma de Correct First Attempt']

# x axis for plotting import numpy as np
import numpy as np
xx = np.stack(i for i in range(len(y)))

# CROSS VALIDATION ANALYSIS. Se determina la cantidad de vecinos a partir de los datos de entrenamiento
from sklearn import neighbors
from sklearn.cross_validation import cross_val_score

for i, weights in enumerate (['uniform', 'distance']):
    total_scores = []
    for n_neighbords in range(1,30): 
        knn = neighbors.KNeighborsRegressor(n_neighbords, weights=weights)
        knn.fit(X,y)
        scores = -cross_val_score(knn, X, y, scoring='neg_mean_absolute_error', cv=10)
        total_scores.append(scores.mean())
    
    plt.plot(range(0,len(total_scores)),total_scores,marker='o',label=weights)
    plt.ylabel('cv score')
plt.legend()
plt.show()

# La construcción del modelo considera el número de vecinos determinados en el bloque anterior (regresión)
n_neighbords= 5 # best parameter (para las datos considerados se puede utilizar 3 y 5)

# Lectura y carga de datos de Prueba
algebra_05_06__Stdnts_test = pd.read_csv('algebra_2005_2006_Student_test.csv',sep=';')
algebra_05_06__Stdnts_test.head()

X_test = algebra_05_06__Stdnts_test[['Cuenta de Step Name','Suma de Step Duration (sec)','Suma de Correct Step Duration (sec)','Suma de Corrects','Suma de Incorrects','Suma de Hints','Duracion Promedia por paso (sec)','Durancion promedia pasos correctos','Duracion promedia de pasos incorrectos','Promedio de pasos resueltos en primer intento']]
y_test = algebra_05_06__Stdnts_test['Suma de Correct First Attempt']

xx = np.stack(i for i in range(len(y_test)))

for i, wights in enumerate (['uniform', 'distance']):
    knn = neighbors.KNeighborsRegressor(n_neighbords, weights=weights)
    y_pred = knn.fit(X,y).predict(X_test)
    
    plt.subplot(2, 1, i + 1)
    plt.plot(xx, y_test, c='k', label='data')
    plt.plot(xx, y_pred, c='g', label='prediction')
    plt.axis('tight')
    plt.legend()
    plt.title("KNeighborsRegressor (k= %i, weights ='%s')" % (n_neighbords, weights))
plt.show()

#Para Generar Archivo CSV de predicción
submission = pd.read_csv('algebra_2005_2006_Student_Submission.csv',sep=';')
submission["Correct_First_Attempt"]=y_pred
#submission
submission.to_csv('C:/Users/Gustavo Balmaceda/Desktop/MASTER 2018/DSI/TRABAJO FINAL/Datos Desarrollo/Doc Modificados/Students/algebra_2005_2006_Student_Submission.csv',index=False)
print ('Verifica el archivo algebra_2005_2006_Student_Submission.csv en el directorio definido')

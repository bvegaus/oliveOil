# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 09:32:01 2018

@author: belen
"""
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


np.random.seed(7)


# =============================================================================
# Datos from 2017-2018 harvest
# =============================================================================
datos = pd.read_excel('datosf4.xlsx')
columnas = ['TIPO L-noL','RIP Height','TIPO Ex-noEx']
datos = datos.drop(columnas, axis = 1)



X = datos.drop('CATEGORÍA', axis = 1)
y = datos['CATEGORÍA']


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state = 40)

# =============================================================================
# NEURAL NETWORK MODEL
# =============================================================================
res={}


def red(neuronas, X_train,X_test,y_train,y_test):
    model = Sequential()
    model.add(Dense(neuronas,input_dim=163, activation = "relu"))
    model.add(Dense(3, activation ="softmax"))

    model.compile(optimizer='Adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train,y_train, epochs=200,verbose=0)

    ## evaluate the model
    scores = model.evaluate(X_test, y_test)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    res[neuronas] = scores[1]*100
    
    

for i in range(3,163):
    red(i,X_train,X_test,y_train,y_test)
    print("Neuronas: %d" % i)
    


valores = res.values()

#
(pd.DataFrame.from_dict(data=res, orient='index')
   .to_csv('dict_file_f4_NonStandard.csv', header=False))
 
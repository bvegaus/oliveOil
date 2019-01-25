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
# D1-D2 Harvest
# =============================================================================
##   Read Data
datos_f3 = pd.read_excel('./datos.xlsx', sheet_name=0)
datos_f2 = pd.read_excel('./datos.xlsx', sheet_name=2)

# Partitioning of the dataset in train/test
train = pd.concat([datos_f3[:327],datos_f2[:234]])
test = pd.concat([datos_f3[327:],datos_f2[234:]])


X_train = train.drop(['Name','Class','Baseline','RIP Position','RIP Height'], axis = 1)
y_train = train.Class
y_train = y_train.map({
       'E': 1,
       'L':0,
       'V': 0
       })


X_test = test.drop(['Name','Class','Baseline','RIP Position','RIP Height'], axis = 1)
y_test = test.Class
y_test = y_test.map({
       'E': 1,
       'L':0,
       'V': 0
       })

# =============================================================================
# NEURAL NETWORK MODEL
# =============================================================================
res={}


def red(neuronas, X_train,X_test,y_train,y_test):
    model = Sequential()
    model.add(Dense(neuronas,input_dim=113, activation = "relu"))
    model.add(Dense(2, activation ="softmax"))

    model.compile(optimizer='Adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train,y_train, epochs=200,verbose=0)

    ## evaluate the model
    scores = model.evaluate(X_test, y_test)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    res[neuronas] = scores[1]*100
    
    

for i in range(2,113):
    red(i,X_train,X_test,y_train,y_test)
    print("Neuronas: %d" % i)
    


valores = res.values()

(pd.DataFrame.from_dict(data=res, orient='index')
   .to_csv('dict_file_f2f3_EVOOnonEVOO_NonStandard.csv', header=False))
 
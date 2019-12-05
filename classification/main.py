# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import clasificacion as cla
from sklearn.preprocessing import StandardScaler


# =============================================================================
#               NO AUTO-SCALING
# =============================================================================


###############################################################################
#   Read Data
datos_f3 = pd.read_excel('./datosf3.xlsx')
datos_f2 = pd.read_excel('./datosf2.xlsx')

#Division of the dataset
train = pd.concat([datos_f3[:327],datos_f2[:234]])
test = pd.concat([datos_f3[327:],datos_f2[234:]])

X_train = train.drop(['Name','Class','Baseline', 'RIP Position', 'RIP Height'], axis = 1)
y_train = train.Class

X_test = test.drop(['Name','Class','Baseline', 'RIP Position', 'RIP Height'], axis = 1)
y_test = test.Class

###############################################################################
#   Classification task
cla.mejor_k(X_train,X_test,y_train,y_test)
cla.knn_TT(X_train,X_test,y_train,y_test,3)
cla.SVM(X_train,X_test,y_train,y_test)
cla.arbol(X_train,X_test,y_train,y_test)
cla.regresor(X_train,X_test,y_train,y_test)
cla.xgbo(X_train,X_test,y_train,y_test)





#%%
# =============================================================================
#               AUTOSCALED
# =============================================================================


###############################################################################
#Auto-Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
###############################################################################
#   Classification task
cla.mejor_k(X_train,X_test,y_train,y_test)
cla.knn_TT(X_train,X_test,y_train,y_test,3)
cla.SVM(X_train,X_test,y_train,y_test)
cla.arbol(X_train,X_test,y_train,y_test)
cla.regresor(X_train,X_test,y_train,y_test)
cla.xgbo(X_train,X_test,y_train,y_test)






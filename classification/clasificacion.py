# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score  
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, cross_val_score



    
def SVM(X_train, X_test, y_train, y_test):
    #Train
    svclassifier = SVC()
    svclassifier.fit(X_train, y_train)
        #Evaluation
    y_pred = svclassifier.predict(X_test)
    print('Accuracy SVM TT:  '+ np.str(accuracy_score(y_test, y_pred)*100))
    
    
    
    
def mejor_k(X_train, X_test, y_train, y_test):
    k_range = range(3,10)
    scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train,y_train)
        scores.append(knn.score(X_test,y_test))
        
    plt.figure()
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.scatter(k_range, scores)
    plt.xticks([0,5,10,15,20])
    
    
def knn(X, y, X_train, X_test, y_train, y_test,n):
    knn = KNeighborsClassifier(n)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    
    print('Accuracy KNN TT:  '+ np.str(accuracy_score(y_test, y_pred)*100))
       
    
def knn_TT(X_train, X_test, y_train, y_test,n):
    knn = KNeighborsClassifier(n)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    
    print('Accuracy KNN TT:  '+ np.str(accuracy_score(y_test, y_pred)*100))
    #print('Accuracy KNN CV:  '+ np.str(cross_val_score(knn,X,y,scoring='accuracy',cv=10)*100))
       
    
    
def arbol(X_train, X_test, y_train, y_test):
    arbol = tree.DecisionTreeClassifier(max_depth=10)
    arbol.fit(X_train,y_train)
    y_pred = arbol.predict(X_test)
    
    print('Accuracy Árbol TT:  '+ np.str(accuracy_score(y_test, y_pred)*100))
    
def regresor(X_train, X_test, y_train, y_test):
    reg = LogisticRegression()
    reg.fit(X_train,y_train)
    y_pred = reg.predict(X_test)
    
    print('Accuracy Regresión TT:  '+ np.str(accuracy_score(y_test, y_pred)*100))

def xgbo(X_train, X_test, y_train, y_test):
    #xg_reg = xgb.XGBClassifier(gamma = 0.5, learning_rate=0.64,object='binary:logistic',subsample=0.5,num_class=3, max_depth=10,booster='gbtree')
    xg_reg = xgb.XGBClassifier()
    xg_reg.fit(X_train, y_train)
    y_pred = xg_reg.predict(X_test)
    
    print('Accuracy XGBoost TT:  '+ np.str(accuracy_score(y_test, y_pred)*100))

    #%%
    


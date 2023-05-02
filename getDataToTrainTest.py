# -*- coding: utf-8 -*-
"""
Created on Tue May  2 11:36:19 2023

@author: david
"""
import pandas as pd
import os
import numpy as np
from sklearn.linear_model  import LinearRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer 

def getDataToTrainTest(path,wantFunc:bool, wantMetrics:bool, wantComplexity:bool, transform=None):
    # TRAIN DATA
    # Join the selected files into 1 and remove duplicated IDs
    listOfCsv=list()
    devPath = path +"\\dev"
    compPath = path +"\\comp"
    
    if wantFunc:
        listOfCsv.append('functions')
    if wantMetrics:
        listOfCsv.append('change_metrics')
    if wantComplexity:
        listOfCsv.append('complexity')
    
    trainData = pd.DataFrame([])
    testData = pd.DataFrame([])
    i=1
    for file in listOfCsv:
        file_train = pd.read_csv(os.path.join(devPath,file+'_dev.csv'))
        file_test= pd.read_csv(os.path.join(compPath,file+'_comp.csv')) 
        if i!=1:
            file_train.drop('functionId',axis=1, inplace=True)
            file_test.drop('functionId',axis=1, inplace=True)
            
        trainData=pd.concat([trainData,file_train],axis=1)
        testData=pd.concat([testData,file_test],axis=1)
        i=0
        
    if "functions" not in listOfCsv:
        file_train = pd.read_csv(os.path.join(devPath,'functions_dev.csv'))  
        trainData=pd.concat([trainData,file_train['bug']],axis=1)
    
    trainData = trainData.replace([np.inf, -np.inf], np.nan)
    testData = testData.replace([np.inf, -np.inf], np.nan)
    
    if trainData.isnull().values.any():
        columns = trainData.columns.to_numpy()
        modelToFit = LinearRegression()
        imp = IterativeImputer(estimator=modelToFit, verbose=2, max_iter=30, tol=1e-10, imputation_order='roman')
        formattedData = imp.fit_transform(trainData.drop(['functionId'],axis=1))
        trainData = pd.concat([trainData['functionId'], pd.DataFrame(formattedData, columns=np.delete(columns,columns=='functionId'))],axis=1)
            
    return trainData, testData
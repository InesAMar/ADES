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
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import statsmodels.api as sm

def getDataToTrainTest(path,wantFunc:bool, wantMetrics:bool, wantComplexity:bool, transform=None):
    # TRAIN DATA
    # Join the selected files into 1 and remove duplicated IDs
    listOfCsv=list()
    devPath = path +"\\dev"
    compPath = path +"\\comp"
    
    dataTested=""
    if wantFunc:
        dataTested+="functions;"
        listOfCsv.append('functions')
    if wantMetrics:
        dataTested+="change_metrics;"
        listOfCsv.append('change_metrics')
    if wantComplexity:
        dataTested+="complexity;"
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


# FEATURE SELECTION ##########################################################

# Backward elimination:
def BackwardElimination(X, y, significance_level=0.05):
    num_features = X.shape[1]
    for i in range(num_features):
        regressor = sm.OLS(y, X).fit()
        max_pvalue = max(regressor.pvalues)
        if max_pvalue > significance_level:
            max_index = np.argmax(regressor.pvalues)
            X = np.delete(X, max_index, axis=1)
        else:
            break    
    return X


# DATA REDUCTION ##########################################################

# PCA:
def PCAfunction(X):

    # create a PCA pipeline  
    pca = PCA()
    pipeline = Pipeline([('pca', pca)])

    # Define parameter grid for grid search
    param_grid = {'pca__n_components': [5, 10, 15, 20, 25]}  # Number of components to try

    # Perform grid search
    grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=5)
    grid_search.fit(X)

    # Get the best PCA model
    best_pca = grid_search.best_estimator_.named_steps['pca']
    best_n_components = best_pca.n_components_

    # Fit the best PCA model on the data
    trainData_pca = best_pca.fit_transform(X)

    # Print the best number of components
    print("Best number of components:", best_n_components)

    return trainData_pca, best_n_components

# LDA:
#def LDAfunction (trainData, n_components):

    # create an LDA object and fit the data
    lda = LinearDiscriminantAnalysis(n_components)
    lda.fit(trainData)

    # transform the data to the new coordinate system
    trainData_lda = lda.transform(trainData) 

    return trainData_lda 

# BORUTA:
#def Borutafunction (trainData):

    # create a random forest classifier
    rf = RandomForestClassifier(n_estimators=10, random_state=42)

    # create a Boruta object and fit the data
    boruta = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=42)
    boruta.fit(trainData)

    # transform the data 
    trainData_boruta = boruta.transform(trainData) 

    return trainData_boruta, boruta.support_


 """
    if transform is not None:
        # z-score transformation
        trainToTransform = trainData.drop(columns=['functionId','bug'])
        
        testToTransform = trainData.drop(columns=['functionId'])
        mean = trainToTransform.mean()
        std_dev = trainToTransform.std()
        trainToTransform = (trainToTransform - mean) / std_dev
        return trainData, testData, dataTested, mean, std_dev
    """
    return trainData, testData, dataTested

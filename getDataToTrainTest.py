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

def getDataToTrainTest(path, wantFunc: bool, wantMetrics: bool, wantComplexity: bool):
    # function that receives the main path and which files to create a dataframe with
    # all the relevant data
    # Specify the list of CSV files to be included based on the provided flags
    listOfCsv = []
    devPath = os.path.join(path, "dev")
    compPath = os.path.join(path, "comp")
    
    # Check which files to concatenate
    dataTested = ""
    if wantFunc:
        dataTested += "functions;"
        listOfCsv.append('functions')
    if wantMetrics:
        dataTested += "change_metrics;"
        listOfCsv.append('change_metrics')
    if wantComplexity:
        dataTested += "complexity;"
        listOfCsv.append('complexity')
    
    # Initialize train and test dataframes
    trainData = pd.DataFrame([])
    testData = pd.DataFrame([])
    i = 1
    
    # Iterate over the selected CSV files and concatenate them into trainData and testData dataframes
    for file in listOfCsv:
        file_train = pd.read_csv(os.path.join(devPath, file + '_dev.csv'))
        file_test = pd.read_csv(os.path.join(compPath, file + '_comp.csv')) 
        
        # Drop 'functionId' column from all files except the first one to avoid duplication
        if i != 1:
            file_train.drop('functionId', axis=1, inplace=True)
            file_test.drop('functionId', axis=1, inplace=True)
        
        # Concatenate files to DataFrame object
        trainData = pd.concat([trainData, file_train], axis=1)
        testData = pd.concat([testData, file_test], axis=1)
        i = 0
    
    # If 'functions' is not included in the selected CSV files, add the 'bug' column from 'functions_dev.csv' to trainData
    if "functions" not in listOfCsv:
        file_train = pd.read_csv(os.path.join(devPath, 'functions_dev.csv'))  
        trainData = pd.concat([trainData, file_train['bug']], axis=1)
    
    # Replace infinite values with NaN
    trainData = trainData.replace([np.inf, -np.inf], np.nan)
    testData = testData.replace([np.inf, -np.inf], np.nan)
    
    # Perform data imputation if there are any missing values in trainData
    # If any NA values are found, implemnt a Linear Regression to create a suitable value
    if trainData.isnull().values.any():
        columns = trainData.columns.to_numpy()
        modelToFit = LinearRegression()
        imp = IterativeImputer(estimator=modelToFit, verbose=2, max_iter=30, tol=1e-10, imputation_order='roman')
        formattedData = imp.fit_transform(trainData.drop(['functionId'], axis=1))
        trainData = pd.concat([trainData['functionId'], pd.DataFrame(formattedData, columns=np.delete(columns, columns=='functionId'))], axis=1)

    return trainData, testData, dataTested

# FEATURE SELECTION ##########################################################
# Backward elimination:
def BackwardElimination(X_train, X_test, outFeatures, y, significance_level=0.05):
    # Get the number of features
    num_features = X_train.shape[1]
    
    # Initialize an array to store the indices of eliminated features
    ind = np.array([])
    
    # Perform backward elimination
    for i in range(num_features):
        # Fit an OLS (Ordinary Least Squares) regression model
        regressor = sm.OLS(y, X_train).fit()
        
        # Find the maximum p-value among the coefficients
        max_pvalue = max(regressor.pvalues)
        
        # Check if the maximum p-value is above the significance level
        if max_pvalue > significance_level:
            # Find the index of the feature with the maximum p-value
            max_index = np.argmax(regressor.pvalues)
            
            # Delete the corresponding feature from the training and testing sets
            X_train = np.delete(X_train, max_index, axis=1)
            X_test = np.delete(X_test, max_index, axis=1)
            
            # Delete the corresponding feature from the list of output features
            outFeatures = np.delete(outFeatures, max_index, axis=1)
            
            # Store the index of the eliminated feature
            ind = np.append(ind, max_index)
        else:
            # Stop the elimination process if all remaining features have p-values below the significance level
            break
    
    # Return the updated training and testing sets, the updated list of output features, and the indices of eliminated features
    return X_train, X_test, outFeatures, ind

# DATA REDUCTION ##########################################################

# PCA:
def PCAfunction(X):

    # create a PCA pipeline  
    pca = PCA()
    pipeline = Pipeline([('pca', pca)])

    # Define parameter grid for grid search
    # Values of components should always be lower the number of features
    param_grid = {'pca__n_components': np.arange(1,X.shape[1],5).tolist()}  # Number of components to try

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

    return trainData_pca, best_n_components, best_pca



# Not in use ##########################################################

# LDA:
def LDAfunction(trainData, n_components):

    # create an LDA object and fit the data
    lda = LinearDiscriminantAnalysis(n_components)
    lda.fit(trainData)

    # transform the data to the new coordinate system
    trainData_lda = lda.transform(trainData) 

    return trainData_lda 

# BORUTA:
def Borutafunction(trainData):

    # create a random forest classifier
    rf = RandomForestClassifier(n_estimators=10, random_state=42)

    # create a Boruta object and fit the data
    boruta = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=42)
    boruta.fit(trainData)

    # transform the data 
    trainData_boruta = boruta.transform(trainData) 

    return trainData_boruta, boruta.support_


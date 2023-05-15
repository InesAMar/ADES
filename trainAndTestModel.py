# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 09:50:25 2023

@author: david
"""

import numpy as np 
from sklearn.metrics import roc_auc_score, f1_score
from typing import Callable, Any
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import AdaBoostClassifier

def trainAndTestModel(typeOfModel:Callable[... , Any], X, y, k_fold, dictArgsClassfier, overOrUnder:Callable[...,Any]=None, dictArgsSamp=None,transform=None):
    # Split the data into train and test sets with equal class proportions
    splitter = StratifiedKFold(n_splits = k_fold)
    
    accuracies=[]
    rocAUCs=[]
    f1Scores=[]
    bestAuc=0.0
    for train_indices, test_indices in splitter.split(X, y):
        
        X_train, y_train = X.loc[train_indices], y.loc[train_indices]
        X_test, y_test = X.loc[test_indices], y.loc[test_indices]
    
        mean = X_train.mean()                           
        std_dev = X_train.std()
        if transform is not None:
            # z-score transformation
            X_train = (X_train - mean) / std_dev
            X_test = (X_test - mean) / std_dev
            
        if overOrUnder is not None:
            rus = overOrUnder(**dictArgsSamp)
    
            X_train, y_train = rus.fit_resample(X_train, y_train)
            
        # define model
        modelToTrain= typeOfModel(**dictArgsClassfier)
        
        # fit model
        modelToTrain.fit(X_train, y_train)
        
        # Test it
        outProbs = modelToTrain.predict_proba(X_test)
        outPrediction = np.argmax(outProbs, axis=1)
        
        accuracy = np.mean(outPrediction == y_test)
        accuracies.append(accuracy)
        
        roc_auc = roc_auc_score(y_test, outPrediction)
        rocAUCs.append(roc_auc)
        if roc_auc>bestAuc:
          bestMean = mean 
          bestStd = std_dev
          bestModel = modelToTrain
          bestAuc = roc_auc

        f1Score = f1_score(y_test, outPrediction)
        f1Scores.append(f1Score)
    
    meanAcc=sum(accuracies)/len(accuracies)
    stdAcc =np.std(accuracies)
    meanRocAuc=sum(rocAUCs)/len(rocAUCs)
    stdRocAuc =np.std(rocAUCs)
    meanf1Score=sum(f1Scores)/len(f1Scores)
    stdf1Score =np.std(f1Scores)
    
    return bestModel, [meanAcc,stdAcc], [meanRocAuc,stdRocAuc], [meanf1Score,stdf1Score], bestMean, bestStd


def trainAndTestEnsembleModel(typeOfModel:Callable[... , Any], X, y, k_fold, dictArgsClassfier, overOrUnder:Callable[...,Any]=None, dictArgsSamp=None):
    # Split the data into train and test sets with equal class proportions
    splitter = StratifiedKFold(n_splits = k_fold)
    
    accuracies=[]
    rocAUCs=[]
    f1Scores=[]
    bestAuc=0.0
    for train_indices, test_indices in splitter.split(X, y):
        
        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = X[test_indices], y[test_indices]
        
        if overOrUnder is not None:
            rus = overOrUnder(**dictArgsSamp)
    
            X_train, y_train = rus.fit_resample(X_train, y_train)
        
        # define model
        modelToTrain= typeOfModel(**dictArgsClassfier)
        
        ensemble_model = AdaBoostClassifier(modelToTrain, n_estimators=10)

        # fit ensemble of models
        ensemble_model.fit(X_train, y_train)
        
        # Test it
        outProbs = ensemble_model.predict_proba(X_test)
        outPrediction = np.argmax(outProbs, axis=1)
        
        accuracy = np.mean(outPrediction == y_test)
        accuracies.append(accuracy)
        
        roc_auc = roc_auc_score(y_test, outPrediction)
        rocAUCs.append(roc_auc)
        if roc_auc>bestAuc:
          bestModel = ensemble_model
          bestAuc= roc_auc

        f1Score = f1_score(y_test, outPrediction)
        f1Scores.append(f1Score)
    
    meanAcc=sum(accuracies)/len(accuracies)
    stdAcc =np.std(accuracies)
    meanRocAuc=sum(rocAUCs)/len(rocAUCs)
    stdRocAuc =np.std(rocAUCs)
    meanf1Score=sum(f1Scores)/len(f1Scores)
    stdf1Score =np.std(f1Scores)
    
    return bestModel, [meanAcc,stdAcc], [meanRocAuc,stdRocAuc], [meanf1Score,stdf1Score]
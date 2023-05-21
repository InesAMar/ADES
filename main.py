# -*- coding: utf-8 -*-
"""
Created on Tue May  2 20:55:57 2023

@author: david
"""
#General Imports
import os 
import pandas as pd
import numpy as np

# Feature understanding 
from sklearn.decomposition import PCA

# Sampler Imports
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import RandomOverSampler


# Classifier Imports
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

#Homemade Imports
from getDataToTrainTest import getDataToTrainTest
from trainAndTestModel import trainAndTestModel2, trainAndTestEnsembleModel
from graphsOfData import graphsOfData

# Paths
path = os.getcwd()
compPath = os.path.join(path,'comp')
devPath = os.path.join(path, 'dev')

# Data Selection, Select which csv's to use (Functions, Metrics, Complexity)
wantFunc = True
wantMetrics = True
wantComplexity = True
trainData, testData, dataTested = getDataToTrainTest(path, wantFunc, wantMetrics, wantComplexity,1)
#testData.to_csv(os.path.join(compPath,'test_mergedfile.csv'), index=False)

graphsOfData(trainData, path = path)

# Parameters for Model Train and Test 
k_fold=5
test_percentage=1/k_fold
#nFeat=20 # For PCA if used 
kSmote=10 



#Data transformation


# Choose Sampling Strategy by choosing the index in "strategy"
strategies=[RandomUnderSampler,
            RandomOverSampler,
            SMOTE,
            ADASYN]

strategy = strategies[0]

sampling_strategy = 1.0 # ratio between classes
if strategy == SMOTE:
    dictArgsSamp={'random_state':42, 
                  'sampling_strategy':sampling_strategy,
                  'k_neighbors': kSmote}
else:
    dictArgsSamp={'random_state':42, 
                  'sampling_strategy':sampling_strategy}


# List of Models To Test
listOfModels = list([ SVC, 
                      KNeighborsClassifier, 
                      RandomForestClassifier,
                      DecisionTreeClassifier, 
                      LogisticRegression,
                      #KMeans,
                      #AgglomerativeClustering
                      ])

for classifier in listOfModels:
    
    if classifier == LogisticRegression:
        listOfArgs = {'penalty': ['l1', 'l2'],
                      'C': [0.1, 1, 0.85,0.7,0.9,0.5],
                      'solver': ['liblinear', 'saga']
                      }
    elif classifier == KNeighborsClassifier:
        listOfArgs =  {
            'n_neighbors': [3, 5, 7, 10, 15],
            'weights': ['uniform', 'distance'],
            'p': [1, 2]}
    elif classifier == SVC:
        listOfArgs = {'C': [0.1, 0.7, 0.9, 1],
                        'kernel': ['linear', 'rbf','sigmoid'],
                        'gamma': [0.1, 0.5, 0.1, 0.01, 'scale'],
                        "probability":[True]}

    elif (classifier == DecisionTreeClassifier) or (classifier == RandomForestClassifier):
        listOfArgs={"max_depth":[2,5,10],"min_samples_leaf":[5,10,15,20,25,30],"criterion":["entropy","log_loss"]}
    else:
        listOfArgs={}
    
    
        
    outData= pd.read_csv(os.path.join(compPath,'test_mergedfile.csv'))
    outPrediction2 = outData['functionId']
    # Train, Test and obtain main results of classifier
    
    pca = True
    backElim = True 
    
    modelToTest, meanRocAUC, meanf1Sco, mean, std, bestParams, pca_object, be_object = trainAndTestModel2(classifier, 
                                                                    trainData.drop(columns=['functionId','bug']), 
                                                                    trainData['bug'],
                                                                    k_fold,
                                                                    listOfArgs,
                                                                    strategy,
                                                                    dictArgsSamp,
                                                                    transform=True,
                                                                    pca=pca,
                                                                    backElim= backElim)
    
    outData=(outData.drop(columns = ['functionId'])-mean)/std
    if pca_object is not None:
        outData2 = pca_object.transform(outData);
    if be_object is not None:
        outData
    
    # Agregate relevant data about the results and process characteristics on dictionary object 
    dataToSave={'Data tested': dataTested,
                'sampling strat':strategy.__name__,
                'kSmote':"notUsed",
                'FeatureTransform': "z-score e PCA",
                'Classifier':type(modelToTest).__name__,	
                'ClassifierHyperP':str(bestParams),
                'F1-measure': meanf1Sco[0],
                'Std F1': meanf1Sco[1],	
                'ROC-AUC': meanRocAUC[0],
                'Std ROC-AUC': meanRocAUC[1]}
    
    # Save the data onto csv file for posterior processing
    dataToSave = pd.DataFrame([dataToSave])
    dataToSave.to_csv(os.path.join(devPath,"pca_zscore_final_results.csv"), mode='a',sep=';', index=False, header=True)

    rowcount=0
    for row in open(os.path.join(devPath,"pca_zscore_final_results.csv")):
      rowcount+= 1
    #printing the result
    print("Number of lines present:-", rowcount) 

    #outData2 = pca.transform(outData.drop(columns = ['functionId']))
    if pca:
        outPrediction = modelToTest.predict(outData2)
    else:
        outPrediction = modelToTest.predict(outData)

    outData['bug'] = outPrediction
    outData['functionId'] = outPrediction2
    outData = outData[['functionId','bug']]
    outData.to_csv(os.path.join(compPath,f'pca_zscore_Try_{rowcount}_predictions.csv'), sep=';', index=False)
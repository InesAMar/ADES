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
# --> Fazer ciclo para testar todos
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

for classifier in listOfModels: # --> testar todos
    
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
        listOfArgs={"max_depth":[2,5,10],
                    "min_samples_leaf":[5,10,15,20,25,30],
                    "criterion":["entropy","log_loss"]}
    else:
        listOfArgs={}
    
    
    outData = pd.read_csv(os.path.join(compPath,'test_mergedfile.csv'))
    outFeatures = outData.drop(columns=['functionId','bug'])
    outPrediction2 = outData['functionId']
    
    # Train, Test and obtain main results of classifier
    
        # --> testar pca sim e não
    # --> testar transform sim e não + testar backElim sim e não

    pca = True
    backElim = True 

    modelToTest, meanRocAUC, meanf1Sco, bestParams, test_features = trainAndTestModel2(classifier, 
                                                                    trainData.drop(columns=['functionId','bug']), 
                                                                    trainData['bug'],
                                                                    outFeatures,
                                                                    k_fold,
                                                                    listOfArgs,
                                                                    strategy,
                                                                    dictArgsSamp,
                                                                    transform=True,
                                                                    pca=pca,
                                                                    backElim= backElim)
    
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
    # --> por tudo no mm ficheiro

    rowcount=0
    for row in open(os.path.join(devPath,"pca_zscore_final_results.csv")):
      rowcount+= 1
    #printing the result
    print("Number of lines present:-", rowcount) 

    #outData2 = pca.transform(outData.drop(columns = ['functionId']))
    newOutData = pd.concat([outData['functionId'], test_features, outData['bug']],axis=1)
    outPrediction = modelToTest.predict(newOutData)

    newOutData['bug'] = outPrediction
    outData2csv = newOutData[['functionId','bug']]
    outData2csv.to_csv(os.path.join(compPath,f'pca_zscore_Try_{rowcount}_predictions.csv'), sep=';', index=False)
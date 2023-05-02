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
from sklearn.tree import DecisionTreeClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

#Homemade Imports
from getDataToTrainTest import getDataToTrainTest
from trainAndTestModel import trainAndTestModel, trainAndTestEnsembleModel
from graphsOfData import graphsOfData

# Paths
path = os.getcwd();
compPath = os.path.join(path,'comp')
devPath = os.path.join(path, 'dev')


# Data Selection, Select which csv's to use (Functions, Metrics, Complexity)
wantFunc = True
wantMetrics = True
wantComplexity = True
trainData, testData = getDataToTrainTest(path, wantFunc, wantMetrics, wantComplexity)

graphsOfData(trainData, path = path)

# Parameters for Model Train and Test 
k_fold=10
test_percentage=1/k_fold
nFeat=20 # For PCA if used 
kSmote=10 # For 

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
                      LogisticRegression])

for classifier in listOfModels:
    
    if classifier == LogisticRegression:
        listOfArgs = list([{"penalty":'l2', "C":0.1},{"penalty":'l2', "C":1},{"penalty":'l2', "C":0.85}])
    elif classifier == KNeighborsClassifier:
        listOfArgs = list([{"n_neighbors": 5},{"n_neighbors": 10},{"n_neighbors": 15}])
    elif classifier == SVC:
        listOfArgs=list([{"kernel":"sigmoid","C":0.7,"probability":True},{"kernel":"sigmoid","C":0.9,"probability":True},
                         {"kernel":"sigmoid","probability":True},
                          {"C":0.7,"probability":True},{"C":0.9,"probability":True},
                         {"probability":True}])
    elif (classifier == DecisionTreeClassifier) or (classifier == RandomForestClassifier):
        listOfArgs=list([{"max_depth":10,"min_samples_leaf":30,"criterion":"entropy"},
                         {"max_depth":10,"min_samples_leaf":30,"criterion":"log_loss"},
                         {"max_depth":10,"min_samples_leaf":30}])
    else:
        listOfArgs=list([])
        
        
    for dictArgsClassfier in listOfArgs:
        outData = testData
        # Train, Test and obtain main results of classifier
        modelToTest, meanAcc, meanRocAUC, meanf1Sco = trainAndTestModel(classifier, 
                                                                        trainData.drop(['functionId','bug']), 
                                                                        trainData['bug'],
                                                                        k_fold,
                                                                        dictArgsClassfier,
                                                                        dictArgsSamp)
        
        # Agregate relevant data about the results and process characteristics on dictionary object 
        dataToSave={'SMOTE k':kSmote,
                    'FeatureSelection': "PCA",
                    'nFeat': nFeat,
                    'Classifier':type(modelToTest).__name__,	
                    'ClassifierHyperP':str(dictArgsClassfier),	
                    'Accuracy': meanAcc,
                    'F1-measure': meanf1Sco,	
                    'ROC-AUC': meanRocAUC}
        
        # Save the data onto csv file for posterior processing
        dataToSave = pd.DataFrame([dataToSave])
        dataToSave.to_csv(os.path.join(devPath,"final_results.csv"), sep=';', mode='a', index=False, header=False)

        rowcount=0
        for row in open(os.path.join(devPath,"final_results.csv")):
          rowcount+= 1
        #printing the result
        print("Number of lines present:-", rowcount) 

        #outData2 = pca.transform(outData.drop(columns = ['functionId']))
        outPrediction = modelToTest.predict(outData.drop(columns = ['functionId']))

        outData['bug'] = outPrediction
        outData = outData[['functionId','bug']]
        outData.to_csv(os.path.join(compPath,f'Try_{rowcount}_predictions.csv'), sep=';', index=False)


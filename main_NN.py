# -*- coding: utf-8 -*-
"""
Created on Tue May  2 20:55:57 2023

@author: david
"""
#General Imports
import os 
import pandas as pd
import numpy as np
import torch

# Sampler Imports
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import RandomOverSampler

#Homemade Imports
from getDataToTrainTest import getDataToTrainTest
from trainAndTestModel import trainAndTestNeuralNetwork
from SmallNetwork import SmallNetwork
from graphsOfData import graphsOfData

# Paths
path = os.getcwd();
compPath = os.path.join(path,'comp')
devPath = os.path.join(path, 'dev')


# Data Selection, Select which csv's to use (Functions, Metrics, Complexity)
 # Select by assigning True or False
wantFunc = True
wantMetrics = True
wantComplexity = True
trainData, testData, dataTested = getDataToTrainTest(path, wantFunc, wantMetrics, wantComplexity,1)
testData.to_csv(os.path.join(compPath,'test_mergedfile.csv'), index=False)


# Get graphs of data for the selected Data
graphsOfData(trainData, path = path)

# Parameters for Model Train and Test 
k_fold=9
test_percentage=1/k_fold
nFeat=20 # For PCA if used 
kSmote=10 # For 

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

listOfNumber = [20, 30]

num_epochs = 100
learning_rate = 0.001
dropout=0.1

for number in listOfNumber:
    classifier = SmallNetwork
    dictArgs= {"inputSize":len(testData.columns)-1,"numClasses": 2,"fc1": number,"fc2": number, "dropout": dropout}
    outData= pd.read_csv(os.path.join(compPath,'test_mergedfile.csv'))
    outPrediction2 = outData['functionId']
    
    # Train, Test and obtain main results of classifier
    transform = True
    modelToTest, meanAcc, meanRocAUC, meanf1Sco, mean, std = trainAndTestNeuralNetwork(classifier,
                                                                                       dictArgs,
                                                                                       num_epochs,
                                                                                       learning_rate,                                        
                                                                                       trainData.drop(columns=['functionId','bug']), 
                                                                                       trainData['bug'],
                                                                                       k_fold,
                                                                                       strategy,
                                                                                       dictArgsSamp,
                                                                                       transform)
        
    
    if transform:
        outData=(outData.drop(columns = ['functionId'])-mean)/std
    # Agregate relevant data about the results and process characteristics on dictionary object 
    dataToSave={'Data tested': dataTested,
                'sampling strat':strategy.__name__,
                'kSmote':"notUsed",
                'num_epochs': num_epochs,
                'n_nodes': number,
                'Classifier':type(modelToTest).__name__,
                'Accuracy': meanAcc[0],
                'Std Acc': meanAcc[1],
                'F1-measure': meanf1Sco[0],
                'Std F1': meanf1Sco[1],	
                'ROC-AUC': meanRocAUC[0],
                'Std ROC-AUC': meanRocAUC[1]}
    
    # Save the data onto csv file for posterior processing
    dataToSave = pd.DataFrame([dataToSave])
    dataToSave.to_csv(os.path.join(devPath,"NN_z-score_final_results.csv"), mode='a',sep=';', index=False, header=True)

    rowcount=0
    for row in open(os.path.join(devPath,"NN_z-score_final_results.csv")):
      rowcount+= 1
    #printing the result
    print("Number of lines present:-", rowcount) 

    #outData2 = pca.transform(outData.drop(columns = ['functionId']))
    
    outData2 = torch.tensor(outData.to_numpy()).float()
    with torch.no_grad():
        outPrediction = modelToTest(outData2)
        _, predicted = torch.max(outPrediction.data, 1)
    
    outData['bug'] = predicted
    outData['functionId'] = outPrediction2
    outData = outData[['functionId','bug']]
    outData.to_csv(os.path.join(compPath,f'NN_z-score_Try{rowcount}_predictions.csv'), sep=';', index=False)


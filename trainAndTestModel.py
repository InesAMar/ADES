# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 09:50:25 2023

@author: david
"""

import numpy as np 
from sklearn.metrics import roc_auc_score, f1_score
from typing import Callable, Any
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from getDataToTrainTest import BackwardElimination, PCAfunction

import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

def trainAndTestModel2(typeOfModel:Callable[... , Any], X, y, outFeatures, k_fold, dictArgsClassifier, overOrUnder:Callable[...,Any]=None, dictArgsSamp=None, transform=None, pca=None, backElim=None):
    # Split the data into train and test sets with equal class proportions
    splitter = StratifiedKFold(n_splits=k_fold)
    
    # Evaluation Metrics and Variables to record data
    rocAUCs=[]
    f1Scores=[]
    bestAuc=0.0
    bestModel = None
    bestParams = None
    
    # Implement a k_fold cross-validation technique
    for train_indices, test_indices in splitter.split(X, y):
        
        X_train, y_train = X.loc[train_indices], y.loc[train_indices]
        X_test, y_test = X.loc[test_indices], y.loc[test_indices]
        outFeatures1 = outFeatures.copy() 
        
        # Implement Feature Selection OR Data Reduction
        # z-score transformation
        mean = X_train.mean()                           
        std_dev = X_train.std()
        
        if transform:
            # Implement z-transform in training and test sets
            X_train = (X_train - mean) / std_dev
            X_test = (X_test - mean) / std_dev
            outFeatures1 = (outFeatures1 - mean) / std_dev
            
        if pca:
            # Implement PCA in training and test sets
            pca_features, nFeat, pca_object = PCAfunction(X_train)
            X_train = pca_features
            X_test = pca_object.transform(X_test)
            outFeatures1 = pca_object.transform(outFeatures1)
            
        if backElim:
            # Implement Backwards Elimination in training and test sets
            if type(X_train)==np.ndarray:
                X_train, X_test, outFeatures1, ind = BackwardElimination(X_train, X_test, outFeatures1, y_train, 0.05)
            else: 
                X_train, X_test, outFeatures1, ind = BackwardElimination(X_train.to_numpy(), X_test.to_numpy(), outFeatures1.to_numpy(), y_train, 0.05)
        
        if overOrUnder is not None:
            # Implement Under or Oversampling of the training set
            rus = overOrUnder(**dictArgsSamp)
            X_train, y_train = rus.fit_resample(X_train, y_train)
        
        # define model
        modelToTrain = typeOfModel()
        
        # Perform grid search with all possible parameter values
        grid_search = GridSearchCV(modelToTrain, param_grid=dictArgsClassifier, cv=k_fold)
        grid_search.fit(X_train, y_train)
        
        # Use the best parameters for training
        bestModel = grid_search.best_estimator_
        bestParams = grid_search.best_params_
        print("Best parameters:", bestParams)
        
        bestParams['nFeat'] = X_train.shape[1]
        # Test the model
        outProbs = bestModel.predict_proba(X_test)
        outPrediction = np.argmax(outProbs, axis=1)
        
        roc_auc = roc_auc_score(y_test, outPrediction)
        rocAUCs.append(roc_auc)
        if roc_auc > bestAuc:
            bestAuc = roc_auc

        f1Score = f1_score(y_test, outPrediction)
        f1Scores.append(f1Score)
    
    meanRocAuc = sum(rocAUCs) / len(rocAUCs)
    stdRocAuc = np.std(rocAUCs)
    meanf1Score = sum(f1Scores) / len(f1Scores)
    stdf1Score = np.std(f1Scores)
    
    return bestModel, [meanRocAuc, stdRocAuc], [meanf1Score, stdf1Score], bestParams, outFeatures1


def trainAndTestNeuralNetwork(model:Callable[... , Any],dictArgs, num_epochs, learning_rate, X, y, k_fold, overOrUnder:Callable[...,Any]=None, dictArgsSamp=None,transform=None):
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
            
        batch_size = 5
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        modelToTrain = model(**dictArgs)
        modelToTrain.to(device)
        # Define the loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(modelToTrain.parameters(), lr=learning_rate)
        
        X_train = torch.tensor(X_train.to_numpy()).float()
        y_train = torch.tensor(y_train.to_numpy()).long()
        X_test = torch.tensor(X_test.to_numpy()).float()
        y_test = torch.tensor(y_test.to_numpy()).long()
        
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Training loop
        for epoch in tqdm.tqdm(range(num_epochs)):
            correct=0
            total=0
            for i, (images, labels) in enumerate(train_loader):
                # Move tensors to the configured device
                images = images.to(device)
                labels = labels.to(device)
        
                # Forward pass
                outputs = modelToTrain(images)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f'epoch {epoch}: Train Accuracy: {(100 * correct / total):.2f}%')


        
        
        # Test the model
        modelToTrain.eval()  # Evaluation mode (e.g., disables dropout)
        outPrediction = np.array([])
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = modelToTrain(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                outPrediction = np.append(outPrediction, predicted.cpu().view(-1))
                
            accuracy = np.mean(outPrediction == y_test)
            accuracies.append(accuracy)
            print(f'Test Accuracy: {(100 * correct / total):.2f}%')
        
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

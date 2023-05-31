# -*- coding: utf-8 -*-
"""
Created on Wed May 31 23:17:24 2023

@author: david
"""

#FOR Feature Importance With Integrated Gradients
from captum.attr import IntegratedGradients
from SmallNetwork import SmallNetwork

# Data access
from getDataToTrainTest import getDataToTrainTest

# Libraries for viewing results
import matplotlib.pyplot as plt
import numpy as np
import os 
import torch
from torch.utils.data import Dataset, Dataloader
from sklearn.model_selection import StratifiedKFold

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = self.X[idx]
        label = self.y[idx]
        return sample,label
    
    
# To Run this model you need to implement first 
# torch.save(modelToTest.state_dict(), 'model.pth')
# in the main_NN script
# and understand in which fold was the model saves
# this is where it obtained the best performance
    
pickModel= 'model.pth' #INSERT MODEL NAME
path =  os.getcwd()
pathModel= os.path.join(path, pickModel)
motor_classes = ['Not a Bug', 'Bug']  

trainData, testData, dataTested = getDataToTrainTest(path, False, False, True)

# Initialize the variables with the approaprite values for the model you will load
num_classes = 2
k_fold = 5

X=trainData.drop(columns=['functionId','bug'])
y=trainData['bug']
splitter = StratifiedKFold(n_splits = k_fold)

foldWhereModelWasSaved = 3

for train_indices, test_indices in splitter.split(X, y):
    X_train, y_train = X.loc[train_indices], y.loc[train_indices]
    X_test, y_test = X.loc[test_indices], y.loc[test_indices]
    break

# Get the test data to implement feature importance
dataset = CustomDataset(X_test.to_numpy(), y_test.to_numpy())
testloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

# introduce the parameters of the model you previously saved
number = 10
number2 = 20
dropout  = 0.1
dictArgs= {"inputSize":len(testData.columns)-1,"numClasses": 2,"fc1": number,"fc2": number2, "dropout": dropout}
# Initialize your model, with Correct model specifications (Check results.csv)
model = SmallNetwork(**dictArgs)

model.load_state_dict(torch.load(pathModel, map_location='cpu'))
model.eval()
# Create an instance of the IntegratedGradients method
ig = IntegratedGradients(model)

feature_names= [X_train.columns[i] for i in range(5)]

plt.figure(figsize = (9, 9))
# Iterate over the test dataset
for classN in range(num_classes):
  feature_importance = np.zeros([1,5])
  for i, data in enumerate(testloader, 0):
      # Get the input and target
      input, target = data[0].float(), data[1].squeeze().long()
      if (target==classN):
        # Compute feature importance for the input of a certain Class
        attributions = ig.attribute(input, target=target)
        feature_importance += attributions.numpy()
      
  # Average the feature importance over the dataset size
  feature_importance /= len(X_test)
  # Create the bar plot
  ax = plt.subplot( num_classes,1, classN+1)
  plt.bar(feature_names, feature_importance.reshape(5))

  # Add title, x-axis label, y-axis label
  title= "Feature Importance for Class = "+motor_classes[classN]
  ax.set_title(title)
  ax.set_ylabel("Importance Value")

  # Add a grid
  ax.grid(True)
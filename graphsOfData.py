# -*- coding: utf-8 -*-
"""
Created on Tue May  2 11:28:55 2023

@author: david
"""

import matplotlib.pyplot as plt
import seaborn as sns

def graphsOfData(trainData, path):
    """ANÁLISE DAS FEATURES"""
    
    plt.figure(figsize = (40, 30))
    
    features_names = trainData.drop(columns = ['functionId','bug']).columns
    nbins=20
    for i in range(len(features_names)):
      
        # Gets the feature name
        feature = features_names[i]
        bin_len= (trainData[feature].max()-trainData[feature].min())
        containers = [bin_len*i/nbins + trainData[feature].min()  for i in range(nbins+1)]
        
        
        # Plots the probability distribution for the class and the rest
        ax = plt.subplot(5, 6, i+1)
        sns.histplot(data = trainData.drop(columns = ['functionId']), x = feature, hue = 'bug', color = 'b', ax = ax, bins = containers)
        ax.set_title(feature)
        ax.legend(["Bug", "Not a Bug"])
        ax.set_xlabel(f"Distribution of {feature}")
    
    
    plt.savefig(path+'/distribuitionClasses.png')
    
    plt.figure(figsize = (40, 30))
    
    features_names = trainData.drop(columns = ['functionId','bug']).columns
    nbins=20
    for i in range(len(features_names)):
      
        # Gets the feature name
        feature = features_names[i]
        class1_data = trainData[trainData['bug'] == 0][feature]
        class2_data = trainData[trainData['bug'] == 1][feature]
        
        # Plots the probability distribution for the class and the rest
        ax = plt.subplot(5, 6, i+1)
        
        ax.boxplot([class1_data, class2_data])
        ax.set_xticklabels(["Not a Bug","Bug"])
        ax.set_xlabel(f"Distribution of {feature}")    
    
    plt.savefig(path+'/boxplots_dist.png')
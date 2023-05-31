# -*- coding: utf-8 -*-
"""
Created on Tue May  2 11:28:55 2023

@author: david
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import SpectralClustering
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from getDataToTrainTest import getDataToTrainTest
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV

def graphsOfData(trainData, path):
    """FEATURE ANALYSIS"""
    
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

    #Heatmaps of Correlation Matrices
    corr_matrix = np.corrcoef(trainData.drop(columns=['functionId','bug']).transpose())

    fig, ax = plt.subplots()
    im = ax.imshow(corr_matrix)
    im.set_clim(-1, 1)
    ax.grid(False)
    ax.set_xlabel("Feature's index")
    ax.set_ylabel("Feature's index")
    cbar = ax.figure.colorbar(im, ax=ax, format='% .2f')
    
    plt.savefig(path+'/corrmatrix.png')

    # Create an instance of KMeans
    X = trainData.drop(columns = ['functionId','bug']).to_numpy()
    y= trainData['bug']
    parameters = {'n_clusters': [2, 3, 4, 5, 6, 7, 8, 9, 10]}
    clustering= KMeans()
    #Implement GridSearch of number of clusters with silhouette scoring technique
    grid_search = GridSearchCV(clustering, parameters, scoring=silhouette_score)
    grid_search.fit(X)
    
    best_n_clusters = grid_search.best_params_['n_clusters']
    best_labels = grid_search.best_estimator_.labels_
    print("Best number of clusters:", best_n_clusters)
    print("Cluster:", best_labels)
    df = pd.DataFrame({'Cluster': best_labels, 'Bug': y})
    # Group the DataFrame by Cluster and calculate the counts of bug and non-bug instances
    grouped_df = df.groupby('Cluster')['Bug'].value_counts().unstack(fill_value=0)
    # Generate bar plot
    grouped_df.plot(kind='bar', stacked=True)
    plt.xlabel('Cluster')
    plt.ylabel('Counts')
    plt.title('Bug and Non-Bug Counts in Each Cluster')
    plt.legend(['Non-Bug', 'Bug'])
    plt.show()



    X = trainData.drop(columns=['functionId', 'bug']).to_numpy()
    y= trainData['bug']
    parameters = {'n_clusters': [2, 3, 4, 5, 6, 7, 8, 9, 10]}
    clustering = AgglomerativeClustering()
    grid_search = GridSearchCV(clustering, parameters, scoring=silhouette_score)
    grid_search.fit(X)
    best_n_clusters = grid_search.best_params_['n_clusters']
    best_labels = grid_search.best_estimator_.labels_
    print("Best number of clusters:", best_n_clusters)
    print("Cluster:", best_labels)
    
    # Create a DataFrame with cluster labels and ground truth labels
    df = pd.DataFrame({'Cluster': best_labels, 'Bug': y})
    # Group the DataFrame by Cluster and calculate the counts of bug and non-bug instances
    grouped_df = df.groupby('Cluster')['Bug'].value_counts().unstack(fill_value=0)
    # Generate bar plot
    grouped_df.plot(kind='bar', stacked=True)
    plt.xlabel('Cluster')
    plt.ylabel('Counts')
    plt.title('Bug and Non-Bug Counts in Each Cluster')
    plt.legend(['Non-Bug', 'Bug'])
    plt.show()
import time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plot_functions as charts
import Normalize as norm
from itertools import cycle, islice
from sklearn import datasets, metrics, cluster, mixture
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.feature_selection import SelectKBest, f_classif
from yellowbrick.cluster import KElbowVisualizer



def run(source,data):

    #data = pd.read_csv('Data/pd_speech_features.csv', sep=',', decimal='.', skiprows=1)    


    data_norm = norm.normalization(source,data)

    if source == "PD":
        target = "class"
    else:
        target = "Cover_Type"

    y = data_norm.pop(target).values
    X = data_norm.values

    kmeans_models = []
    kmeans_inertia = []
    best_kmeans_inertia = float("inf")
    best_k = 1

    
    k_list = [1,2,3,4,5,6,7,8,9,10]    
    for k in k_list:
        kmeans_model = cluster.KMeans(n_clusters=k, random_state=1).fit(X)
        if kmeans_model.inertia_ < best_kmeans_inertia:
            best_kmeans_inertia = kmeans_model.inertia_
            best_k = k
            y_pred = kmeans_model.labels_
        kmeans_inertia.append(kmeans_model.inertia_)

    visualizer = KElbowVisualizer(cluster.KMeans(), k=(4,12))

    visualizer.fit(X)        # Fit the data to the visualizer
    visualizer.show()        # Finalize and render the figure
    



    print("Best k for Clusters :" + str(k))
    print("Sum of squared distances :" + str(best_kmeans_inertia))

    ### Silhouette and Rand index Scores for the best K
    print("Silhouette:",metrics.silhouette_score(X, y_pred))
    print("RI[KMeans] =",adjusted_rand_score(y, y_pred))

    #### Visualize K-MEANS for our dataset
    """
    plt.figure(figsize=(10, 4))

    y_pred_filtered = cluster.KMeans(n_clusters=3,random_state=random_state).fit_predict(X)
    plt.subplot(144)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred_filtered)
    plt.title("Unevenly Sized Blobs")
    plt.show()
    """



    '''
    plt.figure(figsize=(5, 5))
    plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05, hspace=.01)
    color_array = ['#377eb8','#ff7f00','#4daf4a','#f781bf','#a65628','#984ea3','#999999','#e41a1c','#dede00']
    plot_num = 1
    plt.title("KMeans", size=18)

    colors = np.array(list(islice(cycle(color_array),int(max(y) + 1))))
    colors = np.append(colors, ["#000000"]) #black color for outliers (if any)
    plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])
    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)
    plt.xticks(())
    plt.yticks(())
    plt.text(.99, .01, ('%.2fs' % efficiency).lstrip('0'),
        transform=plt.gca().transAxes,size=15,horizontalalignment='right')

    plt.show()'''


def statistics(source,data):

    data_norm = norm.normalization(source,data)

    
    if source == "PD":
        target = "class"
    else:
        target = "Cover_Type"

    y = data_norm.pop(target).values
    X = data_norm.values

    kmeans_model = cluster.KMeans(n_clusters=7, random_state=1).fit(X)
    y_pred = kmeans_model.labels_

    print("Clustering :")
    #print("1.Algorithm : K-means") 
    #print("a) List of k used : [1,2,3,4,5,6,7,8,9,10]") 
    print("a) Number of Clusters : 7 ")
    print("b) Sum of squared distances :", kmeans_model.inertia_)
    print("c) Silhouette:",metrics.silhouette_score(X, y_pred))
    print("d) Rand Index =",adjusted_rand_score(y, y_pred))


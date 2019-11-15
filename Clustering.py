import time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle, islice
from sklearn import datasets, metrics, cluster, mixture
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.feature_selection import SelectKBest, f_classif


def clustering(data: pd.DataFrame):
    dataset = data
    y = dataset.pop('class').values
    X = data.values

    t0  = time.time()
    kmeans_model = cluster.KMeans(n_clusters=3, random_state=1).fit(X)
    y_pred = kmeans_model.labels_
    efficiency = time.time() - t0

    print("Sum of squared distances:",kmeans_model.inertia_)
    print("Silhouette:",metrics.silhouette_score(X, y_pred))
    print("RI[KMeans] =",adjusted_rand_score(y, y_pred))
    
    plt.figure(figsize=(7,5))
    #plt.subplot(141)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred)
    plt.title("KMeans")
    plt.show()
   # X_new = SelectKBest(f_classif, k=10).fit_transform(X, y_pred)
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
    return 0
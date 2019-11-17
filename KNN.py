import numpy as np

import pandas as pd
from pandas.plotting import register_matplotlib_converters

import matplotlib.pyplot as plt

import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier

import plot_functions as func

def simple_knn(data):

    y: np.ndarray = data.pop('class').values
    X: np.ndarray = data.values
    labels = pd.unique(y)

    trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)

    nvalues = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 25, 31, 35, 40, 45, 50, 52, 55, 60, 70, 80, 90, 100, 105, 110, 120, 130, 140, 150]
    dist = ['manhattan', 'euclidean', 'chebyshev']
    values = {}
    for d in dist:
        yvalues = []
        for n in nvalues:
            knn = KNeighborsClassifier(n_neighbors=n, metric=d)
            knn.fit(trnX, trnY)
            prdY = knn.predict(tstX)
            yvalues.append(metrics.accuracy_score(tstY, prdY))
        values[d] = yvalues

    plt.figure()
    func.multiple_line_chart(plt.gca(), nvalues, values, 'KNN variants', 'n', 'accuracy', percentage=True)
    plt.show()

def k_near_ngb(trnX, tstX, trnY, tstY, labels, plot):
    register_matplotlib_converters()

    """
    print(data.shape)

    y: np.ndarray = data.pop('class').values
    X: np.ndarray = data.values
    labels = pd.unique(y)

    trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)
    """
    nvalues = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 25, 31, 35, 40, 45, 50, 52, 55, 60, 70, 80, 90, 100, 105, 110, 120, 130, 140, 150]
    dist = ['manhattan', 'euclidean', 'chebyshev']

    
    acc_values = {}
    sens_values = {}
    max_accuracy = 0
    max_sensitivity = 0


    for d in dist:
        accuracy_values = []
        sensitivity_values = []

        for n in nvalues:
            knn = KNeighborsClassifier(n_neighbors=n, metric=d)
            knn.fit(trnX, trnY)
            prdY = knn.predict(tstX)
            accuracy = metrics.accuracy_score(tstY, prdY)
            accuracy_values.append(accuracy)

            tn, fp, fn, tp = metrics.confusion_matrix(tstY, prdY, labels).ravel()
            sensitivity = tn/(tn+fp)
            sensitivity_values.append(sensitivity)

            cnf_mtx = metrics.confusion_matrix(tstY, prdY, labels)
            
            if accuracy > max_accuracy:
                best_accuracy = [(d, n), accuracy, sensitivity, cnf_mtx]
                max_accuracy = accuracy
                
            if sensitivity > max_sensitivity:
                best_sensitivity = [(d, n), accuracy, sensitivity, cnf_mtx]
                max_sensitivity = sensitivity
        
        acc_values[d] = accuracy_values
        sens_values[d] = sensitivity_values



    if plot:
        plt.figure()
        func.multiple_line_chart(plt.gca(), nvalues, acc_values, 'KNN variants', 'n', 'accuracy', percentage=True)
        plt.show()
        func.multiple_line_chart(plt.gca(), nvalues, sens_values, 'KNN variants', 'n', 'sensitivity', percentage=True)
        plt.show()

    return ["kNN", best_accuracy, best_sensitivity]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.ensemble import GradientBoostingClassifier
import plot_functions as func


def gradient_boost(trnX, tstX, trnY, tstY, labels, plot):

    lr_list = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]
    n_estimators = [5, 10, 25, 50, 75, 100, 150, 200, 250, 300]
    max_depths = [5, 10, 25, 50]
    max_features = ['sqrt', 'log2']

    max_accuracy = 0
    max_sensitivity = 0


    plt.figure()
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), squeeze=False)
    for k in range(len(max_features)):
        f = max_features[k]
        acc_values = {}
        sens_values = {}
        for d in max_depths:
            accuracy_values = []
            sensitivity_values = []
            for n in n_estimators:
                max_learning_accuracy = 0

                for l in lr_list:
                    gb = GradientBoostingClassifier(n_estimators=n, learning_rate=l, max_depth=d, max_features=f)
                    gb.fit(trnX, trnY)
                    prdY = gb.predict(tstX)

                    accuracy = metrics.accuracy_score(tstY, prdY)

                    tn, fp, fn, tp = metrics.confusion_matrix(tstY, prdY, labels).ravel()
                    sensitivity = tn/(tn+fp)
                    sensitivity_values.append(sensitivity)

                    cnf_mtx = metrics.confusion_matrix(tstY, prdY, labels)

                    if accuracy > max_accuracy:
                        best_accuracy = [(f, d, n, l), accuracy, sensitivity, cnf_mtx]
                        max_accuracy = accuracy
                
                    if sensitivity > max_sensitivity:
                        best_sensitivity = [(f, d, n, l), accuracy, sensitivity, cnf_mtx]
                        max_sensitivity = sensitivity
                    
                    if accuracy > max_learning_accuracy:
                        max_learning_accuracy = accuracy 
                
                accuracy_values.append(max_learning_accuracy)

            acc_values[d] = accuracy_values
            sens_values[d] = sensitivity_values

                
                
        func.multiple_line_chart(axs[0, k], n_estimators, acc_values, 'Gradient Boost with %s features'%f, 'nr estimators', 
                                 'accuracy', percentage=True)

    if plot:
        plt.show()
        

    return ["Gradient Boost", best_accuracy, best_sensitivity]


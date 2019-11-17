import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import sklearn.metrics as metrics
import plot_functions as func


def xg_boost(trnX, tstX, trnY, tstY, labels, plot):
    

    n_estimators = [5, 10, 25, 50, 75, 100, 150, 200, 250, 300]
    max_depths = [5, 10, 25, 50]

    max_accuracy = 0
    max_sensitivity = 0


    plt.figure()
    
    acc_values = {}
    sens_values = {}
    
    for d in max_depths:
        accuracy_values = []
        sensitivity_values = []
        for n in n_estimators:
            xgb = XGBClassifier(n_estimators=n, max_depth=d)
            xgb.fit(trnX, trnY)
            prdY = xgb.predict(tstX)

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

                
        func.multiple_line_chart(plt.gca(), n_estimators, acc_values, 'XG Boost', 'nr estimators', 
                                 'accuracy', percentage=True)

    if plot:
        plt.show()
        

    return ["XGBoost", best_accuracy, best_sensitivity]
    


import pandas as pd
import numpy as np
from pandas.plotting import register_matplotlib_converters

import data_balancing as balance
import Normalize as norm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

import KNN as knn
import decision_tree as dt
import naive_bayes as nb
import RandomForest as rf
import GradientBoost as gb
import XGBoost as xgb

import print_statistics as stats



def preprocessing(trnX, tstX, trnY, tstY, balance_data, normalize_data, strategy):
    pass


def unsupervised(data):
    pass


def classification(data):
    
    # split data set in target variable and atributes
    y: np.ndarray = data.pop('class').values
    X: np.ndarray = data.values
    labels: np.ndarray = pd.unique(y)

    # store accuracies and sensitivities for each classifier
    accuracys = {"nb":[], "knn":[], "dt":[], "rf":[], "gb":[], "xgb":[]}
    specificities = {"nb":[], "knn":[], "dt":[], "rf":[], "gb":[], "xgb":[]}
    

    cv = KFold(n_splits=10, random_state=42, shuffle=False)

    for train_index, test_index in cv.split(X):
        
        trnX, tstX, trnY, tstY = X[train_index], X[test_index], y[train_index], y[test_index]
        
        # normalize and balance the dataset
        trnX, tstX, trnY, tstY = norm.standardScaler(trnX, tstX, trnY, tstY)
        trnX, trnY = balance.run(trnX, trnY, 'all', 42, False)
        
        # classify with fixed parameters and get the metrics
        acc = [0,0,0,0,0,0]
        spec = [0,0,0,0,0,0]
        acc[0], spec[0], cnf_mtx = nb.simple_naive_bayes(trnX, tstX, trnY, tstY, labels)
        acc[1], spec[1] = knn.simple_knn(trnX, tstX, trnY, tstY, 1, 'manhattan', labels)
        acc[2], spec[2] = dt.simple_decision_tree(trnX, tstX, trnY, tstY, 0.05, 5, 'entropy', labels)
        acc[3], spec[3] = rf.simple_random_forest(trnX, tstX, trnY, tstY, 150, 10, 'sqrt', labels)
        acc[4], spec[4] = gb.simple_gradient_boost(trnX, tstX, trnY, tstY, 100, 0.1, 5, 'sqrt', labels)
        acc[5], spec[5] = xgb.simple_xg_boost(trnX, tstX, trnY, tstY, 200, 5, labels)
        
        # store metrics
        i = 0
        for clf in accuracys:
            accuracys[clf] += [acc[i]]
            specificities[clf] += [spec[i]]
            i += 1
        print("1")
    

    # calculate avg accuracy and avg specificity 
    avg_accuracys = []
    avg_specificities = []
    for clf in accuracys:
        avg_acc_clf = sum(accuracys[clf])/len(accuracys[clf])
        avg_accuracys += [avg_acc_clf]

        avg_spec_clf = sum(specificities[clf])/len(specificities[clf])
        avg_specificities += [avg_spec_clf]
    
    # create report for each classifier with struct [clf_name, params, acc, spec, cnf_mtx]
    nb_report = ["Naive Bayes", ('GaussianNB'), avg_accuracys[0], avg_specificities[0], cnf_mtx]
    knn_report = ["kNN", ('manhattan', 1), avg_accuracys[1], avg_specificities[1], cnf_mtx]
    dt_report = ["Decision Tree", ('entropy', 5, 0.05), avg_accuracys[2], avg_specificities[2], cnf_mtx]
    rf_report = ["Random Forest", ('sqrt', 10, 150), avg_accuracys[3], avg_specificities[3], cnf_mtx]
    gb_report = ["Gradient Boosting", ('sqrt', 5, 100, 0.1), avg_accuracys[4], avg_specificities[4], cnf_mtx]
    xgb_report = ["XGBoost", (5, 200), avg_accuracys[5], avg_specificities[5], cnf_mtx]

    reports = [nb_report, knn_report, dt_report, rf_report, gb_report, xgb_report]

    stats.print_report(reports, (True, True))





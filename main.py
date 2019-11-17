import pandas as pd
import numpy as np
from pandas.plotting import register_matplotlib_converters

import data_balancing as balance
import normalize as normalize
from sklearn.model_selection import train_test_split

import KNN as knn
import decision_tree as dt
import naive_bayes as nb
import RandomForest as rf
import GradientBoost as gb
import XGBoost as xgb

import print_statistics as stats



def preprocessing(data, normalize_data, balance_data, strategy):

    if normalize_data:
        data = normalize.run(data) 

    # Strategies: 'minority', 'not majority', 'not minority', 'all' 
    if balance_data:
        data = balance.run(data, strategy, 42, False) 
    
    return data


def unsupervised(data):
    pass


def classification(data, preproces_params):

    # separates attributes from target class
    y: np.ndarray = data.pop('class').values
    X: np.ndarray = data.values
    labels: np.ndarray = pd.unique(y)
    trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)

    nb_report = nb.naive_bayes(trnX, tstX, trnY, tstY, labels, False)
    """knn_report = knn.k_near_ngb(trnX, tstX, trnY, tstY, labels, False)
    dt_report = dt.decision_tree(trnX, tstX, trnY, tstY, labels, False, False)
    rf_report = rf.random_forest(trnX, tstX, trnY, tstY, labels, False)
    gb_report = gb.gradient_boost(trnX, tstX, trnY, tstY, labels, False)
    xgb_report = xgb.xg_boost(trnX, tstX, trnY, tstY, labels, False)"""

    #reports = [nb_report, knn_report, dt_report, rf_report, gb_report, xgb_report]
    reports = [nb_report]
    stats.print_stats(reports, preproces_params)


data = pd.read_csv('Data/pd_speech_features.csv', sep=',', decimal='.', skiprows=1)


type = "classification"

if type == "classification":
    balanced = True
    normalized = True 
    data = preprocessing(data, balanced, normalized, 'all')
    classification(data, (balanced, normalized))





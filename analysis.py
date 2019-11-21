import pandas as pd
import numpy as np

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




def classification(data, analysis):

    balanced = True
    normalized = True 

    # get 1000 samples per class and get new data set

    # separates the dataset in training and testing sets
    y: np.ndarray = data.pop('class').values
    X: np.ndarray = data.values
    labels: np.ndarray = pd.unique(y)

    trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)

    # normalize and balance the dataset
    trnX, tstX, trnY, tstY = norm.standardScaler(trnX, tstX, trnY, tstY)
    trnX, trnY = balance.run(trnX, trnY, 'all', 42, False)

    # find best classifier
    #nb_report = nb.naive_bayes(trnX, tstX, trnY, tstY, labels, False)
    #knn_report = knn.k_near_ngb(trnX, tstX, trnY, tstY, labels, True)
    dt_report = dt.decision_tree(trnX, tstX, trnY, tstY, labels, False, False)
    #rf_report = rf.random_forest(trnX, tstX, trnY, tstY, labels, False)
    #gb_report = gb.gradient_boost(trnX, tstX, trnY, tstY, labels, False)
    #xgb_report = xgb.xg_boost(trnX, tstX, trnY, tstY, labels, False)

    reports = [dt_report]
    #reports = [nb_report, knn_report, dt_report, rf_report, gb_report, xgb_report]

    stats.print_analysis(reports, (balanced, normalized))



def produce_analysis():

    data = pd.read_csv('Data/pd_speech_features.csv', sep=',', decimal='.', skiprows=1)

    classification(data, True)


produce_analysis()
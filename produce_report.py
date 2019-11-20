import pandas as pd
import numpy as np
from pandas.plotting import register_matplotlib_converters

import data_balancing as balance
import normalize as normalize
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
    
    df_trnX = pd.DataFrame(trnX, columns = data.columns)
    df_trnY = pd.DataFrame(trnY, columns = ['class'])
    df_trn = pd.concat([df_trnX, df_trnY], axis=1, sort=False)

    df_tstX = pd.DataFrame(tstX, columns = data.columns)
    df_tstY = pd.DataFrame(tstY, columns = ['class'])
    df_tst = pd.concat([df_tstX, df_tstY], axis=1, sort=False)


    if normalize_data:
        # normalize training set
        df_trn = normalize.normalization(df_trn)
        min_trn_values = df_trn.min().tolist()
        max_trn_values = df_trn.max().tolist()

        # subtract minimum of each variable in training set
        df_trn = df_trn - min_trn_values

        # scale testing set
        idx_col = 0
        len_col = len(df_tst['class'])
        for col in df_tst:
            for i in range(len_col):
                df_tst.loc[i,col] = (df_tst.loc[i,col] - min_trn_values[idx_col]) / max_trn_values[idx_col]
            idx_col += 1
        

    # Strategies: 'minority', 'not majority', 'not minority', 'all' 
    if balance_data:
        df_trn = balance.run(df_trn, strategy, 42, False) 
        

    trnY = df_trn.pop('class').values
    trnX = df_trn.values

    tstY = df_tst.pop('class').values
    tstX = df_tst.values
    
    return trnX, tstX, trnY, tstY


def unsupervised(data):
    pass


def classification(data, preproces_params, analysis):

    if analysis:

        data = normalize.scaler(data)
        data = balance.run(data, 'all', 42, False) 

        # separates the dataset in training and testing sets
        y: np.ndarray = data.pop('class').values
        X: np.ndarray = data.values
        labels: np.ndarray = pd.unique(y)
        trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)

        nb_report = nb.naive_bayes(trnX, tstX, trnY, tstY, labels, False)
        knn_report = knn.k_near_ngb(trnX, tstX, trnY, tstY, labels, True)
        dt_report = dt.decision_tree(trnX, tstX, trnY, tstY, labels, False, False)
        rf_report = rf.random_forest(trnX, tstX, trnY, tstY, labels, False)
        gb_report = gb.gradient_boost(trnX, tstX, trnY, tstY, labels, False)
        xgb_report = xgb.xg_boost(trnX, tstX, trnY, tstY, labels, False)

        #reports = [nb_report, knn_report]
        reports = [nb_report, knn_report, dt_report, rf_report, gb_report, xgb_report]

        stats.print_stats(reports, preproces_params)

    else:
        y: np.ndarray = data.pop('class').values
        X: np.ndarray = data.values

        scores = {"nb":[], "knn":[], "dt":[], "rf":[], "gb":[], "xgb":[]}
        cv = KFold(n_splits=10, random_state=42, shuffle=False)

        for train_index, test_index in cv.split(X):
            print("Train Index: ", train_index, "\n")
            print("Test Index: ", test_index)

            trnX, tstX, trnY, tstY = X[train_index], X[test_index], y[train_index], y[test_index]

            nb_score = nb.simple_naive_bayes(trnX, tstX, trnY, tstY)
            knn_score = knn.simple_knn(trnX, tstX, trnY, tstY, 1, 'manhattan')
            dt_score = dt.simple_decision_tree(trnX, tstX, trnY, tstY, 0.05, 20, 'entropy')
            rf_score = rf.simple_random_forest(trnX, tstX, trnY, tstY, 150, 10, 'sqrt')
            gb_score = gb.simple_gradient_boost(trnX, tstX, trnY, tstY, 100, 0.1, 5, 'sqrt')
            xgb_score = xgb.simple_xg_boost(trnX, tstX, trnY, tstY, 200, 5)

            scores["nb"] += [nb_score]
            scores["knn"] += [knn_score]
            scores["dt"] += [dt_score]
            scores["rf"] += [rf_score]
            scores["gb"] += [gb_score]
            scores["xgb"] += [xgb_score]
        
        print(scores)

        avg_accuracys = []
        for clf in scores:
            avg_score_clf = sum(scores[clf])/len(scores[clf])
            avg_accuracys += [avg_score_clf]
        
        print(avg_accuracys)




def produce_report():

    data = pd.read_csv('Data/pd_speech_features.csv', sep=',', decimal='.', skiprows=1)

    type = "classification"

    if type == "classification":
        
        balanced = True
        normalized = True 

        #trnX, tstX, trnY, tstY, labels = preprocessing(data, balanced, normalized, 'all')

        classification(data, (balanced, normalized), False)





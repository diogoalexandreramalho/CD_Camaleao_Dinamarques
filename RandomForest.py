import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestClassifier
import plot_functions as func


def random_forest(data:pd.DataFrame):
    data: pd.DataFrame = data
    y: np.ndarray = data.pop('class').values
    X: np.ndarray = data.values
    labels = pd.unique(y)

    trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)

    n_estimators = [5, 10, 25, 50, 75, 100, 150, 200, 250, 300]
    max_depths = [5, 10, 25, 50]
    max_features = ['sqrt', 'log2']

    plt.figure()
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), squeeze=False)
    for k in range(len(max_features)):
        f = max_features[k]
        values = {}
        for d in max_depths:
            yvalues = []
            for n in n_estimators:
                rf = RandomForestClassifier(n_estimators=n, max_depth=d, max_features=f)
                rf.fit(trnX, trnY)
                prdY = rf.predict(tstX)
                yvalues.append(metrics.accuracy_score(tstY, prdY))
            values[d] = yvalues
        func.multiple_line_chart(axs[0, k], n_estimators, values, 'Random Forests with %s features'%f, 'nr estimators', 
                                 'accuracy', percentage=True)

    plt.show()
    return 0


def best_randomForest(data: pd.DataFrame, n_est, max_feat, max_dept):
    data: pd.DataFrame = data
    y: np.ndarray = data.pop('class').values
    X: np.ndarray = data.values
    labels = pd.unique(y)

    trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)
    rf = RandomForestClassifier(n_estimators=n_est, max_depth=max_dept, max_features=max_feat)
    rf.fit(trnX, trnY)
    predictions = rf.predict(tstX)
    
    print("Confusion Matrix:")
    print(metrics.confusion_matrix(tstY, predictions))

    print("Classification Report")
    print(metrics.classification_report(tstY, predictions))
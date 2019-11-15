import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
import plot_functions as func


def gradient_boost(data: pd.DataFrame, learning_rate: float):
    data: pd.DataFrame = data
    y: np.ndarray = data.pop('class').values
    X: np.ndarray = data.values
    labels = pd.unique(y)

    trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)

    lr_list = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]
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
                    gb = GradientBoostingClassifier(n_estimators=n, learning_rate=0.05, max_depth=d, max_features=f)
                    gb.fit(trnX, trnY)
                    prdY = gb.predict(tstX)
                    yvalues.append(accuracy_score(tstY, prdY))
            values[d] = yvalues
                #for l in lr_list:
                #    gb = GradientBoostingClassifier(n_estimators=n, learning_rate=l, max_depth=d, max_features=f)
                #    gb.fit(trnX, trnY)
                #    prdY = gb.predict(tstX)
                #    yvalues.append(accuracy_score(tstY, prdY))
                #values[d] = yvalues
        func.multiple_line_chart(axs[0, k], n_estimators, values, 'Gradient Boost with %s features'%f, 'nr estimators', 
                                 'accuracy', percentage=True)

    plt.show()
    return 0


def best_boost(data: pd.DataFrame, n_est, lear_rate, max_feat, max_dept):
    data: pd.DataFrame = data
    y: np.ndarray = data.pop('class').values
    X: np.ndarray = data.values
    labels = pd.unique(y)

    trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)
    gb_clf2 = GradientBoostingClassifier(n_estimators=n_est, learning_rate=lear_rate, max_features=max_feat, max_depth=max_dept, random_state=0)
    gb_clf2.fit(trnX, trnY)
    predictions = gb_clf2.predict(tstX)
    
    print("Confusion Matrix:")
    print(confusion_matrix(tstY, predictions))

    print("Classification Report")
    print(classification_report(tstY, predictions))

    return 0

'''
for learning_rate in lr_list:
    gb_clf = GradientBoostingClassifier(n_estimators=20, learning_rate=learning_rate, max_features=2, max_depth=2, random_state=0)
    gb_clf.fit(trnX, trnY)

    print("Learning rate: ", learning_rate)
    print("Accuracy score (training): {0:.3f}".format(gb_clf.score(trnX, trnY)))
    print("Accuracy score (validation): {0:.3f}".format(gb_clf.score(tstX, tstY)))

gb_clf2 = GradientBoostingClassifier(n_estimators=20, learning_rate=0.5, max_features=2, max_depth=2, random_state=0)
gb_clf2.fit(trnX, trnY)
predictions = gb_clf2.predict(tstX)

print("Confusion Matrix:")
print(confusion_matrix(tstY, predictions))

print("Classification Report")
print(classification_report(tstY, predictions))
'''
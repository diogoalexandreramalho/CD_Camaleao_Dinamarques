import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import plot_functions as func
from subprocess import call
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

def decision_tree(data: pd.DataFrame):
    data: pd.DataFrame = data
    y: np.ndarray = data.pop('class').values
    X: np.ndarray = data.values
    labels = pd.unique(y)

    trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)

    min_samples_leaf = [.05, .025, .01, .0075, .005, .0025, .001]
    max_depths = [5, 10, 25, 50, 100, 200, 400]
    criteria = ['entropy', 'gini']

    plt.figure()
    fig, axs = plt.subplots(1, 2, figsize=(16, 4), squeeze=False)
    for k in range(len(criteria)):
        f = criteria[k]
        values = {}
        for d in max_depths:
            yvalues = []
            for n in min_samples_leaf:
                tree = DecisionTreeClassifier(min_samples_leaf=n, max_depth=d, criterion=f)
                tree.fit(trnX, trnY)
                prdY = tree.predict(tstX)
                yvalues.append(metrics.accuracy_score(tstY, prdY))
            values[d] = yvalues
        func.multiple_line_chart(axs[0, k], min_samples_leaf, values, 'Decision Trees with %s criteria'%f, 'nr estimators', 
                                 'accuracy', percentage=True)

    #plt.show()

    tree = DecisionTreeClassifier(max_depth=3)
    tree.fit(trnX, trnY)

    dot_data = export_graphviz(tree, out_file='dtree.dot', filled=True, rounded=True, special_characters=True)  
    # Convert to png
    from subprocess import call
    call(['dot', '-Tpng', 'dtree.dot', '-o', 'dtree.png', '-Gdpi=600'])

    plt.figure(figsize = (14, 18))
    plt.imshow(plt.imread('dtree.png'))
    plt.axis('off')
    plt.show()

    return 0



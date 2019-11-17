import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import plot_functions as func
from subprocess import call
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

def decision_tree(trnX, tstX, trnY, tstY, labels, plot, png):

    min_samples_leaf = [.05, .025, .02, .015, .01, .0075, .005, .0025, .001]
    max_depths = [5, 10, 25, 50, 100, 200, 400]
    criteria = ['entropy', 'gini']

    max_accuracy = 0
    max_sensitivity = 0
    

    plt.figure()
    fig, axs = plt.subplots(1, 2, figsize=(16, 4), squeeze=False)
    for k in range(len(criteria)):
        f = criteria[k]
        acc_values = {}
        sens_values = {}
        for d in max_depths:
            accuracy_values = []
            sensitivity_values = []
            for n in min_samples_leaf:
                tree = DecisionTreeClassifier(min_samples_leaf=n, max_depth=d, criterion=f)
                tree.fit(trnX, trnY)
                prdY = tree.predict(tstX)

                # accuracy for criteria = f, max_depth = d, min_samples_leaf = n
                accuracy = metrics.accuracy_score(tstY, prdY)
                accuracy_values.append(accuracy)

                # sensitivity for criteria = f, max_depth = d, min_samples_leaf = n
                tn, fp, fn, tp = metrics.confusion_matrix(tstY, prdY, labels).ravel()
                sensitivity = tn/(tn+fp)
                sensitivity_values.append(sensitivity)

                cnf_mtx = metrics.confusion_matrix(tstY, prdY, labels)

                if accuracy > max_accuracy:
                    best_accuracy = [(f, d, n), accuracy, sensitivity, cnf_mtx]
                    max_accuracy = accuracy
                
                if sensitivity > max_sensitivity:
                    best_sensitivity = [(f, d, n), accuracy, sensitivity, cnf_mtx]
                    max_sensitivity = sensitivity

            acc_values[d] = accuracy_values
            sens_values[d] = sensitivity_values

        func.multiple_line_chart(axs[0, k], min_samples_leaf, acc_values, 'Decision Trees with %s criteria'%f, 'nr estimators', 
                                 'accuracy', percentage=True)



    if plot:
        plt.show()


    if png:
        tree = DecisionTreeClassifier(min_samples_leaf=best_accuracy[2], max_depth=best_accuracy[1], criterion=best_accuracy[0])

        dot_data = export_graphviz(tree, out_file='dtree.dot', filled=True, rounded=True, special_characters=True)  
        # Convert to png
        call(['dot', '-Tpng', 'dtree.dot', '-o', 'dtree.png', '-Gdpi=600'])

        plt.figure(figsize = (14, 18))
        plt.imshow(plt.imread('dtree.png'))
        plt.axis('off')
        plt.show()


    return ["Decision Tree", best_accuracy, best_sensitivity]



import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
import plot_functions as plot_funcs

def simple_naive_bayes(trnX, tstX, trnY, tstY, labels):
    
    nb = GaussianNB()
    nb.fit(trnX, trnY)
    prdY = nb.predict(tstX)
    accuracy = metrics.accuracy_score(tstY, prdY)

    tn, fp, fn, tp = metrics.confusion_matrix(tstY, prdY, labels).ravel()
    specificity = tp/(tp+fn)

    return accuracy, specificity


def naive_bayes(trnX, tstX, trnY, tstY, labels, plot):
    register_matplotlib_converters()

    estimators = {'GaussianNB': GaussianNB(), 
                'MultinomialNB': MultinomialNB(), 
                'BernoulyNB': BernoulliNB()}

    xvalues = []
    accuracy_values = []
    specificity_values = []
    max_accuracy = 0
    max_specificity = 0
    
    for clf in estimators:
        xvalues.append(clf)
        estimators[clf].fit(trnX, trnY)
        prdY = estimators[clf].predict(tstX)
        
        accuracy = metrics.accuracy_score(tstY, prdY)
        accuracy_values.append(accuracy)

        tn, fp, fn, tp = metrics.confusion_matrix(tstY, prdY, labels).ravel()
        specificity = tp/(tp+fn)
        specificity_values.append(specificity)

        cnf_mtx = metrics.confusion_matrix(tstY, prdY, labels)

        if accuracy > max_accuracy:
            best_accuracy = [(clf), accuracy, specificity, cnf_mtx]
            max_accuracy = accuracy
        if specificity > max_specificity:
            best_specificity = [(clf), accuracy, specificity, cnf_mtx]
            max_specificity = specificity
    
    
    if plot:
        plt.figure()
        two_series = {'accuracy': accuracy_values, 'sensitivity': specificity_values}
        
        plot_funcs.multiple_bar_chart(plt.gca(), xvalues, two_series, '', '', 'percentage', True)
        
        
        plot_funcs.bar_chart(plt.gca(), xvalues, accuracy_values, 'Comparison of Naive Bayes Models', '', 'accuracy', percentage=True)
        plot_funcs.bar_chart(plt.gca(), xvalues, specificity_values, 'Comparison of Naive Bayes Models', '', 'specifivity', percentage=True)
        plt.show()

    return ["Naive Bayes", best_accuracy, best_specificity]




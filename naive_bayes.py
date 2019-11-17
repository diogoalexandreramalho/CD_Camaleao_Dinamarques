import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
import plot_functions as plot_funcs

def naive_bayes(trnX, tstX, trnY, tstY, labels, plot):
    register_matplotlib_converters()

    """
    # separates attributes from target class
    y: np.ndarray = data.pop('class').values
    X: np.ndarray = data.values
    labels: np.ndarray = pd.unique(y)


    trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)
    """

    estimators = {'GaussianNB': GaussianNB(), 
                'MultinomialNB': MultinomialNB(), 
                'BernoulyNB': BernoulliNB()}

    xvalues = []
    accuracy_values = []
    sensitivity_values = []
    max_accuracy = 0
    max_sensitivity = 0
    
    for clf in estimators:
        xvalues.append(clf)
        estimators[clf].fit(trnX, trnY)
        prdY = estimators[clf].predict(tstX)
        
        accuracy = metrics.accuracy_score(tstY, prdY)
        accuracy_values.append(accuracy)

        tn, fp, fn, tp = metrics.confusion_matrix(tstY, prdY, labels).ravel()
        sensitivity = tn/(tn+fp)
        sensitivity_values.append(sensitivity)

        cnf_mtx = metrics.confusion_matrix(tstY, prdY, labels)

        if accuracy > max_accuracy:
            best_accuracy = [(clf), accuracy, sensitivity, cnf_mtx]
            max_accuracy = accuracy
        if sensitivity > max_sensitivity:
            best_sensitivity = [(clf), accuracy, sensitivity, cnf_mtx]
            max_sensitivity = sensitivity
    
    
    if plot:
        plt.figure()
        two_series = {'accuracy': accuracy_values, 'sensitivity': sensitivity_values}
        
        plot_funcs.multiple_bar_chart(plt.gca(), xvalues, two_series, '', '', 'percentage', True)
        
        
        plot_funcs.bar_chart(plt.gca(), xvalues, accuracy_values, 'Comparison of Naive Bayes Models', '', 'accuracy', percentage=True)
        plot_funcs.bar_chart(plt.gca(), xvalues, sensitivity_values, 'Comparison of Naive Bayes Models', '', 'sensitivity', percentage=True)
        plt.show()

    return ["Naive Bayes", best_accuracy, best_sensitivity]




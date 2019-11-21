import numpy as np
import pandas as pd
import math
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import plot_functions as func

from pandas.plotting import register_matplotlib_converters
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, RandomOverSampler

import data_balancing as balance
import Normalize as norm
import naive_bayes as nb
import KNN as knn
import decision_tree as dt

from sklearn.datasets import make_classification
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.feature_selection import SelectKBest, SelectPercentile, SelectFpr
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel



def balance_analysis(dataset, park):
    data = dataset.copy()
    #Prepare data
    if(park): #Pd_dataset
        data_class = data['class']  #Save the class for later
        data.pop("id") 
        y = data.pop('class').values
        X = data.values

    else: #Cover_Type
        data_class = data['Cover_Type']
        y = data.pop('Cover_Type')
        X = data.values


    classifiers = ["Naive_Bayes", "KNN", "Decision_Tree"]
    balance = ["SMOTE", "None"]
    accuracy = []
    specificity = []

    labels: np.ndarray = pd.unique(y)
    trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y, random_state=41)

    print("Balance Performance")
    for clf in classifiers:
        for bal in balance:
            acc, spec, conf = classify_balance(trnX, tstX, trnY, tstY, labels, clf, bal)
            
            accuracy.append(acc)
            specificity.append(spec)
            
        performance_balance(accuracy, specificity, clf)
        accuracy.clear()
        specificity.clear()

def normalize_analysis(dataset, park):
    data = dataset.copy()
    #Prepare data
    if(park): #Pd_dataset
        data_class = data['class']  #Save the class for later
        data.pop("id") 
        y = data.pop('class').values
        X = data.values

    else: #Cover_Type
        data_class = data['Cover_Type']
        y = data.pop('Cover_Type')
        X = data.values


    classifiers = ["Naive_Bayes", "KNN", "Decision_Tree"]
    normalizers = ["minMaxScaler", "standardScaler", "None"]
    accuracy = []
    specificity = []

    labels: np.ndarray = pd.unique(y)
    trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y, random_state=41)

    print("Normalization Performance")
    for clf in classifiers:
        for n in normalizers:
            acc, spec, conf = classify_normalization(trnX, tstX, trnY, tstY, labels, clf, n)
            
            accuracy.append(acc)
            specificity.append(spec)
            
        performance_normalization(accuracy, specificity, clf)
        accuracy.clear()
        specificity.clear()

def feature_selection_analysis(dataset, park):
    data = dataset.copy()
    #Prepare data
    if(park): #Pd_dataset
        data_class = data['class']  #Save the class for later
        data.pop("id") 
        y = data.pop('class').values
        X = data.values

    else: #Cover_Type
        data_class = data['Cover_Type']
        y = data.pop('Cover_Type')
        X = data.values


    # This computes fval, and pval for every feature 
    fval, pval = f_classif(X, y)

    classifiers = ["Naive_Bayes", "KNN"]
    accuracy = []
    specificity = []
    result_features = {
        "SelectKBest": [0,0],
        "SelectPercentile": [0,0],
        "Wrapper": [0,0]
        }
    
    result_perf = {
        "SelectKBest": [0,0],
        "SelectPercentile": [0,0],
        "Wrapper": [0,0]
        }

    #
    #                      S E L E C T  K  B E S T 
    #
        
    arg = math.floor(math.log(X.shape[1], 2))
    n_features= 2**np.arange(arg+1) # [1,2,4,8,16 ...]
    n_features = np.append(n_features, [X.shape[1]], axis=0)

    
    #Computing sequential foward selections for classifier
    print("Performance of SelectKBest")
    for clf in classifiers:
        for k in n_features:  
            selector = SelectKBest(f_classif, k=k) #k=2 means, I only pick 2 best features
            X_new = selector.fit_transform(X, y)

            #apply classifier -> Returns accuracy and specificity
            acc, spec, conf = classify_feature_selection(X_new, y, clf)
            accuracy.append(acc)
            specificity.append(spec)

        #plot performance graphs
        plot_feature_selection(n_features, accuracy, specificity, clf, "SelectKBest")
        performance_feature_selection(accuracy, specificity, n_features, clf, "SelectKBest", result_features, result_perf)
        
        accuracy.clear()
        specificity.clear()
        
    #
    #                      S E L E C T   P E R C E N T I L E 
    #

    percentile = np.arange(1,11)*10
    accuracy.clear()
    specificity.clear()

    print("Performance of SelectPercentile")
    for clf in classifiers:
        for perc in percentile:  
            selector = SelectPercentile(f_classif, percentile=perc)
            X_new = selector.fit_transform(X, y)

            #apply classifier -> Returns accuracy and specificity
            acc, spec, conf = classify_feature_selection(X_new, y, clf)
            accuracy.append(acc)
            specificity.append(spec)
   
        #plot performance graphs
        plot_feature_selection(percentile, accuracy, specificity, clf, "SelectPercentile")
        performance_feature_selection(accuracy, specificity, percentile, clf, "SelectPercentile", result_features, result_perf)
        
        accuracy.clear()
        specificity.clear()    

    #
    #                           M O D E L   B A S E D 
    #

    #Phase - 1 
    clf1 = ExtraTreesClassifier(n_estimators=50)
    clf1 = clf1.fit(X, y)
    model = SelectFromModel(clf1, prefit=True)
    X_new = model.transform(X)  # -> Chooses the best features

    number_features = X_new.shape[1]
        
    print("Performance of Wrapper")
    for clf in classifiers:
        #Phase - 2 
        #apply classifier -> Returns accuracy and specificity
        acc, spec, conf = classify_feature_selection(X_new, y, clf)
        accuracy.append(acc)
        specificity.append(spec)

        performance_feature_selection(accuracy, specificity, number_features, clf, "Wrapper", result_features, result_perf)

    #Average of features according with bayes and knn
    for feat in result_features.values():
        feat[0] = feat[0]//2 
        feat[1] = feat[1]//2   
    
    for perf in result_perf.values():
        perf[0] = perf[0]/2 
        perf[1] = perf[1]/2   
    
    compareFeatures(result_features)
    comparePerformance(result_perf)
    
    return

def classify_balance(trnX, tstX, trnY, tstY, labels, clf, bal):
    
    trnX, tstX, trnY, tstY = norm.standardScaler(trnX, tstX, trnY, tstY)
    
    if (bal == "SMOTE"):
        trnX, trnY = balance.run(trnX, trnY, 'all', 42, False)
    
    elif (bal == "None"):
        #Does nothing
        pass
    
    #classify
    if (clf == "Naive_Bayes"):
        return nb.simple_naive_bayes(trnX, tstX, trnY, tstY, labels)
    
    elif (clf == "KNN"):
        return knn.simple_knn(trnX, tstX, trnY, tstY, 1, "manhattan", labels)

    elif (clf == "Decision_Tree"):
        return dt.simple_decision_tree(trnX, tstX, trnY, tstY, 0.05, 5, 'entropy', labels)

def classify_normalization(trnX, tstX, trnY, tstY, labels, clf, n):
    
    if (n == "minMaxScaler"):
        trnX, tstX, trnY, tstY = norm.minMaxScaler(trnX, tstX, trnY, tstY)
    
    elif (n == "standardScaler"):
        trnX, tstX, trnY, tstY = norm.standardScaler(trnX, tstX, trnY, tstY)
    
    elif (n == "None"):
        #Does nothing
        pass

    trnX, trnY = balance.run(trnX, trnY, 'all', 42, False)

    #classify
    if (clf == "Naive_Bayes"):
        return nb.simple_naive_bayes(trnX, tstX, trnY, tstY, labels)
    
    elif (clf == "KNN"):
        return knn.simple_knn(trnX, tstX, trnY, tstY, 1, "manhattan", labels)

    elif (clf == "Decision_Tree"):
        return dt.simple_decision_tree(trnX, tstX, trnY, tstY, 0.05, 5, 'entropy', labels)

def classify_feature_selection(X, y, clf):
    
    labels: np.ndarray = pd.unique(y)
    trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y, random_state=41)
    
    # normalize and balance the dataset
    trnX, tstX, trnY, tstY = norm.standardScaler(trnX, tstX, trnY, tstY)
    trnX, trnY = balance.run(trnX, trnY, 'all', 42, False)

    if (clf == "Naive_Bayes"):
        return nb.simple_naive_bayes(trnX, tstX, trnY, tstY, labels)
    
    elif (clf == "KNN"):
        return knn.simple_knn(trnX, tstX, trnY, tstY, 1, "manhattan", labels)

def performance_balance(accuracy, specificity, clf):
    print("SMOTE | None")
    print("\t Using Classifier:" + clf)
    print("\t\t Accuracy: ", accuracy)
    print("\t\t Specificity: ", specificity)

def performance_normalization(accuracy, specificity, clf):
    print("minMaxScaler | standardScaler | none")
    print("\t Using Classifier:" + clf)
    print("\t\t Accuracy: ", accuracy)
    print("\t\t Specificity: ", specificity)

def performance_feature_selection(accuracy, specificity, features, clf, algorithm, result_features, result_perf):
    max_idx_accu = np.argmax(accuracy)
    max_idx_sens = np.argmax(specificity)
    
    if (algorithm == "SelectKBest" or algorithm == "SelectPercentile"):
        n_best_accu = features[max_idx_accu]
        n_best_sens = features[max_idx_accu]

    accuracy = np.round(accuracy, 3)
    specificity = np.round(specificity, 3)

    print("\t Using Classifier: " + clf)
    print("\t\t Accuracy: ", accuracy)
    print("\t\t Specificity: ", specificity)

    if (algorithm == "SelectKBest"):
        print("\t\t Best accuracy: ", accuracy[max_idx_accu], "with n_features = ", n_best_accu)
        print("\t\t Best Specificity: ", specificity[max_idx_sens], "with n_features = ", n_best_sens)

        result_features["SelectKBest"][0] += n_best_accu
        result_features["SelectKBest"][1] += n_best_sens

        result_perf["SelectKBest"][0] += accuracy[max_idx_accu]
        result_perf["SelectKBest"][1] += specificity[max_idx_sens]

    elif (algorithm == "SelectPercentile"):
        print("\t\t Best accuracy: ", accuracy[max_idx_accu], "with percentage of features = ", n_best_accu , "%")
        print("\t\t Best Specificity: ", specificity[max_idx_sens], "with percentage of features = ", n_best_sens , "%")
        result_features["SelectPercentile"][0] += n_best_accu
        result_features["SelectPercentile"][1] += n_best_sens

        result_perf["SelectPercentile"][0] += accuracy[max_idx_accu]
        result_perf["SelectPercentile"][1] += specificity[max_idx_sens]


    elif(algorithm == "Wrapper"):
        print("\t\tBest accuracy: ", accuracy[max_idx_accu], "with n_features = ", features)
        print("\t\tBest Specificity: ", specificity[max_idx_sens], "with n_features = ", features)
        result_features["Wrapper"][0] += features
        result_features["Wrapper"][1] += features

        result_perf["Wrapper"][0] += accuracy[max_idx_accu]
        result_perf["Wrapper"][1] += specificity[max_idx_sens]        

def plot_feature_selection(xaxis, y1axis, y2axis, clf, algorithm):

    plt.subplot(2, 1, 1)
    plt.plot(xaxis, y1axis, 'o-')
    plt.title('Accuracy and Specificity of ' + clf)

    if(algorithm == "SelectKBest"):
        plt.xlabel('n_features')
    
    elif(algorithm == "SelectPercentile"):
        plt.xlabel('Percentage of n_features')
    
    plt.ylabel('accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(xaxis, y2axis, '.-')
    plt.xlabel('n_features')
    plt.ylabel('Specificity')

    plt.show()

    return 0

def compareFeatures(result):
    print("Comparing Features: SelectKBest (n_features) | SelectPercentile (%_features) | Wrapper (n_features) |")
    r = ''
    for feat in result.values():
        r += str(feat[0]) + " | "
    print(r)

def comparePerformance(result):
    print("Comparing Performance [Accuracy, Specificity]: SelectKBest | SelectPercentile | Wrapper |")
    r = ''
    for perf in result.values():
        r += str(perf) + " | "
    print(r)


#Testing
#dataset = pd.read_csv('Data/pd_speech_features.csv', sep=',', decimal='.', skiprows=1)

def preprocessing(dataset, name):
    normalize_analysis(dataset, name)
    balance_analysis(dataset, name)
    feature_selection_analysis(dataset, name)
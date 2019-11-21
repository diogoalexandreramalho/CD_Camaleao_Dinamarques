import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.feature_selection import SelectKBest, SelectPercentile, SelectFpr
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split

import data_balancing as balance
import Normalize as norm
import naive_bayes as nb
import KNN as knn

def feature_selection(data, park):
    #Prepare data
    if(park): #Pd_dataset   
    
        data_class = data['class']  #Save the class for later
        data.pop("id") # Remove id, because we don't care, maybe remove at the begining
        y = data.pop('class').values
        X = data.values

        features_list = data.columns.copy()
        selector = SelectPercentile(f_classif, percentile=55)
        X_new = selector.fit_transform(X, y)

        mask = selector.get_support()
        X_new_data = pd.DataFrame(X_new)

        #This gets me the an array with the names of the features that were selected
        new_features = [] # The list of your K best features
        for bool, feature in zip(mask, features_list):
            if bool:
               new_features.append(feature)

    
        #create new dataset
        new_data = X_new_data.set_axis(new_features, axis=1, inplace=False)
        new_data['class'] = y

        return new_data

    else: #Cover_Type

        data_class = data['Cover_Type']
        y = data.pop('Cover_Type')
        X = data.values

        features_list = data.columns.copy()

        return 
    

#If pd = true, it means that is pd dataset, else is covtype
def feature_selection_analysis(data, park):
    
    #Prepare data
    if(park): #Pd_dataset
        data_class = data['class']  #Save the class for later
        data.pop("id") # Remove id, because we don't care, maybe remove at the begining
        y = data.pop('class').values
        X = data.values

    else: #Cover_Type
        data_class = data['Cover_Type']
        y = data.pop('Cover_Type')
        X = data.values


    # This computes fval, and pval for every feature -> maybe we want to see it
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
            acc, sens, conf = classify(X_new, y, clf)
            accuracy.append(acc)
            specificity.append(sens)

        #plot performance graphs
        plot(n_features, accuracy, specificity, clf, "SelectKBest")
        performance(accuracy, specificity, n_features, clf, "SelectKBest", result_features, result_perf)
        
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
            acc, sens, conf = classify(X_new, y, clf)
            accuracy.append(acc)
            specificity.append(sens)
   
        #plot performance graphs
        plot(percentile, accuracy, specificity, clf, "SelectPercentile")
        performance(accuracy, specificity, percentile, clf, "SelectPercentile", result_features, result_perf)
        
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
        acc, sens, conf = classify(X_new, y, clf)
        accuracy.append(acc)
        specificity.append(sens)

        performance(accuracy, specificity, number_features, clf, "Wrapper", result_features, result_perf)

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

def classify(X, y, clf): # Este data que esta aqui e um np.array -_-
    
    labels: np.ndarray = pd.unique(y)
    trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)
    
    # normalize and balance the dataset
    trnX, tstX, trnY, tstY = norm.standardScaler(trnX, tstX, trnY, tstY)
    trnX, trnY = balance.run(trnX, trnY, 'all', 42, False)

    if (clf == "Naive_Bayes"):
        return nb.simple_naive_bayes(trnX, tstX, trnY, tstY, labels)
    
    elif (clf == "KNN"):
        return knn.simple_knn(trnX, tstX, trnY, tstY, 1, "manhattan", labels)


def plot(xaxis, y1axis, y2axis, clf, algorithm):

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

def performance(accuracy, specificity, features, clf, algorithm, result_features, result_perf):
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
#dataset = pd.read_csv('covtype.data', sep=',', decimal='.')
#data = dataset.copy()
#feature_selection_analysis(data, True)
#new_data = feature_selection(data, True).copy()
#print(new_data)
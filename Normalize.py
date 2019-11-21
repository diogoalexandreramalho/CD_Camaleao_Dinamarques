import pandas as pd
import numpy as np
from pandas.plotting import register_matplotlib_converters
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import data_balancing as balance
import naive_bayes as nb
import KNN as knn
import Decision_Tree as dt

def standardScaler(trnX, tstX, trnY, tstY):

    scaler = StandardScaler()
    trnX = scaler.fit_transform(trnX) 
    tstX = scaler.transform(tstX)
    
    return trnX, tstX, trnY, tstY


def minMaxScaler(trnX, tstX, trnY, tstY):

    scaler = MinMaxScaler()
    trnX = scaler.fit_transform(trnX) 
    tstX = scaler.transform(tstX)
    
    return trnX, tstX, trnY, tstY

def normalize_analysis(data, park):
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


    classifiers = ["Naive_Bayes", "KNN", "Decision_Tree"]
    normalizers = ["minMaxScaler", "standardScaler", "None"]
    accuracy = []
    specificity = []

    labels: np.ndarray = pd.unique(y)
    trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y, random_state=41)

    print("Normalization Performance")
    for clf in classifiers:
        for n in normalizers:
            acc, spec, conf = classify(trnX, tstX, trnY, tstY, labels, clf, n)
            
            accuracy.append(acc)
            specificity.append(spec)
            
        performance(accuracy, specificity, clf)
        accuracy.clear()
        specificity.clear()

def classify(trnX, tstX, trnY, tstY, labels, clf, n): # Este data que esta aqui e um np.array -_-    
    
    if (n == "minMaxScaler"):
        trnX, tstX, trnY, tstY = minMaxScaler(trnX, tstX, trnY, tstY)
    
    elif (n == "standardScaler"):
        trnX, tstX, trnY, tstY = standardScaler(trnX, tstX, trnY, tstY)
    
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

def performance(accuracy, specificity, clf):
    print("minMaxScaler | standardScaler | none")
    print("\t Using Classifier:" + clf)
    print("\t\t Accuracy: ", accuracy)
    print("\t\t Specificity: ", specificity)

#Testing

#dataset = pd.read_csv('Data/pd_speech_features.csv', sep=',', decimal='.', skiprows=1)
#data = dataset.copy()
#normalize_analysis(data, True)


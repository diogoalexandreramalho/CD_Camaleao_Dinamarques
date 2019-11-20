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

import naive_bayes as nb


#If pd = true, it means that is pd dataset, else is covtype
# Nao preciso de dois ifs gigante sporque vou estar a repetir codigo
def feature_selection(data, park):
    
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

    #Classifiers
    classifiers = ["knn", "bayes", "decision", "gradient", "random"]
    accuracy = []
    sensitivy = []

    #
    #                      S E L E C T  K  B E S T 
    #
        
    arg = math.floor(math.log(X.shape[1], 2))
    n_features= 2**np.arange(arg+1) # [1,2,4,8,16 ...]
    n_features = np.append(n_features, [X.shape[1]], axis=0) #-3 Due to the selectKest
    '''
    Nota: O meu numero de features esta a a crescer exponencialmente, se isto der esquesito
            se calhar e melhor trocar
    
    '''
        
    #Computing sequential foward selections for each classifier
    for clf in classifiers:
        for k in n_features:  
            selector = SelectKBest(f_classif, k=k) #k=2 means, I only pick 2 best features
            X_new = selector.fit_transform(X, y)

            #Add class again
            new_data = np.c_[ X_new, data_class.values]
            #printComparison(X, new_data)
                
            #apply classifier -> retorna accuracy e sensitivity e outras cenas
            #acc, sens = classify(new_data, clf)
            #accuracy.append(acc)
            #sensitivy.append(sens)
                
        #plot stuff
        accuracy = [1,2,3,4,5,6, 7, 8, 9, 10, 11] # Apagar isto quando classifier estiver bem
        sensitivy = [1,2,3,4,5,6, 7, 8, 9, 10, 11] # Apagar isto quando classifier estiver bem
        #plot(n_features, accuracy, sensitivy, clf)
        '''
        Nota: Com isto devo conseguir saber a performance para todos os classificadores
        '''            

        
    #
    #                      S E L E C T   P E R C E N T I L E 
    #

    percentile = np.arange(1,11)*10
    accuracy.clear()
    sensitivy.clear()

    for clf in classifiers:
        for perc in percentile:  
            selector = SelectPercentile(f_classif, percentile=perc)
            X_new = selector.fit_transform(X, y)

            #Add class again
            new_data = np.c_[ X_new, data_class.values]
            #printComparison(X, new_data)
            
            #apply classifier -> retorna accuracy e sensitivity e outras cenas
            #acc, sens = classify(new_data, clf)
            #accuracy.append(acc)
            #sensitivy.append(sens)

        #plot stuff
        accuracy = [1,2,3,4,5,6, 7, 8, 9, 10] # Apagar isto quando classifier estiver bem
        sensitivy = [1,2,3,4,5,6, 7, 8, 9, 10] # Apagar isto quando classifier estiver bem
        #plot(percentile, accuracy, sensitivy, clf)

    #
    #                           M O D E L   B A S E D 
    #

    #Phase - 1 
    clf1 = ExtraTreesClassifier(n_estimators=50)
    clf1 = clf1.fit(X, y)
    model = SelectFromModel(clf1, prefit=True)
    X_new = model.transform(X)  # -> Chooses the best features
    
    #print("Feature ranking =",clf.feature_importances_)
    #print("Original data shape:",X.shape,"\nNew data shape:",X_new.shape)
            
    #Add class again
    new_data = np.c_[ X_new, data_class.values]
    #printComparison(X_new, new_data)
    
    #Phase - 2 
    for clf in classifiers:
                
        #apply classifier -> retorna accuracy e sensitivity e outras cenas
        #acc, sens = classify(new_data, clf)
        #accuracy.append(acc)
        #sensitivy.append(sens)

        acc = 1
        sens = 1
        #printPerfomance(acc, sens, clf)

    return

def classify(data, classifier):
    
    # separates the dataset in training and testing sets
    y: np.ndarray = data.pop('class').values
    X: np.ndarray = data.values
    labels: np.ndarray = pd.unique(y)
    trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)
    
    if(classifier == "knn"):
        return knn(data)

    elif(classifier == "bayes"):
        return nb.simple_naive_bayes(trnX, tstX, trnY, tstY, labels)

    elif(classifier == "decision"):
        return decision(data)

    elif(classifier == "gradient"):
        return gradient(data)

    elif(classifier == "random"):
        return random(data)


def plot(xaxis, y1axis, y2axis, clf):

    plt.subplot(2, 1, 1)
    plt.plot(xaxis, y1axis, 'o-')
    plt.title('Accuracy and Sensitivy of ' + clf)
    plt.xlabel('n_features')
    plt.ylabel('accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(xaxis, y2axis, '.-')
    plt.xlabel('n_features')
    plt.ylabel('sensitivity')

    plt.show()

    return 0

def printComparison(old, new):
    print("Old Data:\n", old, "\nNew Data:\n", new)
    print("Size of old:\n", old.shape, "\nSize of new:\n", new.shape)

def printPerfomance(accuracy, sensitivity, clf):
    print("Performance of Classifier: ", clf)
    print("Accuracy: ", accuracy)
    print("Sensitivy: ", sensitivity)



#Testing

dataset = pd.read_csv('pd_speech_features.csv', sep=',', decimal='.', skiprows=1)
#dataset = pd.read_csv('covtype.data', sep=',', decimal='.')
data = dataset.copy()
feature_selection(data, True)
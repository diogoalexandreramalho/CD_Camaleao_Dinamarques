import arff #from liac-arff package
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from IPython.display import display, HTML
from sklearn.preprocessing import LabelBinarizer #for dummification
from mlxtend.frequent_patterns import apriori, association_rules #for ARM 


# The set of regressors that will be tested sequentially
# y the data matrix
def arm(data: pd.DataFrame):
    data: pd.DataFrame = data
    y: np.ndarray = data.pop('class').values
    X: np.ndarray = data.values
    labels = pd.unique(y)

    trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)

    selector = SelectKBest(f_classif, k=10)
    selector.fit(X, y)
    mask = selector.get_support()

    kb = SelectKBest(f_classif, k=10).fit_transform(X, y)
    X_new = pd.DataFrame(kb)


    #This gets me the an array with the names of the features that were selected
    features_list = data.columns.copy()
    new_features = [] # The list of your K best features
    for bool, feature in zip(mask, features_list):
        if bool:
           new_features.append(feature)
    

    new_data = X_new.set_axis(new_features, axis=1, inplace=False)

    print(new_data)
    
    #Discretize
    newdf = new_data.copy()
    for col in newdf:
        newdf[col] = pd.cut(newdf[col], 3, labels=['0','1','2'])

    print(newdf)
#
#
    #Dummify
    dummylist = []
    for att in newdf:
        dummylist.append(pd.get_dummies(newdf[[att]]))
#
    dummified_df = pd.concat(dummylist, axis=1)
    print(dummified_df)
#
    minsup = 0.35 #you can also use iteratively decreasing support as in the previous example
    frequent_itemsets = apriori(dummified_df, min_support=minsup, use_colnames=True)
    print(frequent_itemsets)
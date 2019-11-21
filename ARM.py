import arff #from liac-arff package
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from IPython.display import display, HTML
from sklearn.preprocessing import LabelBinarizer #for dummification
from mlxtend.frequent_patterns import apriori, association_rules #for ARM 
import matplotlib.pyplot as plt
import plot_functions as func
import data_cleaning as cleaner

# The set of regressors that will be tested sequentially
# y the data matrix
def run(source,data):

    data = pd.read_csv('Data/pd_speech_features.csv', sep=',', decimal='.', skiprows=1)    

    if source == "PD":
        target = "class"
    else:
        target = "Cover_Type"
    
    y = data.pop(target).values
    
    dic = cleaner.general_dic(False)
    mfcc_data = cleaner.get_data_from_dic2(data,dic,"Baseline Features")


    X: np.ndarray = data.values
    labels = pd.unique(y)

    #trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)
    
    """
    # Select the best 10 features that describe the model
    selector = SelectKBest(f_classif, k=10)
    kb = selector.fit_transform(X,y)

    new_data = pd.DataFrame(kb)

    # Get Features Selected Names
    features_selected_bool = selector.get_support()
    i = 0
    features_selected_list = []
    for col in data.columns:
        if features_selected_bool[i] == True:
            features_selected_list.append(col)
        i+=1
    
    print("Features Selected : \n")
    for feat in features_selected_list:
        print(feat)
    print("\n")
    """


    #Discretize  - Divide the real variables into three different groups (1,2,3)
    newdf = mfcc_data.copy()
    for col in newdf:
        newdf[col] = pd.cut(newdf[col], 2, labels=['0','1'])

    #
    #
    #Dummify - for each nominal variable , create additional variables for each possible nominal value
    dummylist = []
    for att in newdf:
        dummylist.append(pd.get_dummies(newdf[[att]]))
    dummified_df = pd.concat(dummylist, axis=1)

    print(dummified_df)

    """
    Line graph com numero de patterns, avg quality , avg quality, top n rules, rules por support 

    """
    avg_confidence_list = []
    avg_lift_list = []
    avg_leverage_list = []
    number_of_rules = []
    
    #Multiple Line Chart
    minsup_list = [0.65, 0.65, 0.75, 0.85, 0.95]
    for sup in minsup_list:
        frequent_itemsets = apriori(dummified_df, min_support=sup, use_colnames=True)
        minconf = 0.8
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=minconf)
        rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
        print(rules[(rules['antecedent_len']>=2)])
    
        print(rules)

        confidence = rules["confidence"].values
        lift = rules["lift"].values
        leverage = rules["leverage"].values

        avg_confidence = 0
        avg_lift = 0
        avg_leverage = 0

        for i in range(len(confidence)):
            print(confidence[i])
            avg_confidence += confidence[i]
            avg_lift += lift[i]
            avg_leverage += leverage[i]

        avg_confidence /= len(confidence)
        avg_lift /= len(lift)
        avg_leverage /= len(leverage)
        

        avg_confidence_list.append(avg_confidence)
        avg_lift_list.append(avg_lift)
        avg_leverage_list.append(avg_leverage)
        number_of_rules.append(rules.shape[0])

    plt.figure(figsize=(12,4))
    series = { 'leverage': avg_leverage_list, 'avg_confidence': avg_confidence_list, 'avg_lift': avg_lift_list}
    func.multiple_line_chart(plt.gca(), minsup_list, series, 'Rules Quality', 'date', '')
    plt.show()
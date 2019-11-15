import pandas as pd
import numpy as np
from pandas.plotting import register_matplotlib_converters

import data_balancing as balance
import normalize as normalize
import decision_tree as dt
import naive_bayes as nb
#import Feature_Selection as filt



def preprocessing(data, normalize_data, balance_data, strategy):

    if normalize_data:
        data = normalize.run(data) 

    # Strategies: 'minority', 'not majority', 'not minority', 'all' 
    if balance_data:
        data = balance.run(data, strategy, 42, False) 
    
    return data


def unsupervised(data):
    pass


def classification(data):
    nb.naive_bayes(data)
    #dt.decision_tree(data)


data = pd.read_csv('Data/pd_speech_features.csv', sep=',', decimal='.', skiprows=1)


type = "classification"

if type == "classification":
    data = preprocessing(data, True, True, 'all')

    classification(data)





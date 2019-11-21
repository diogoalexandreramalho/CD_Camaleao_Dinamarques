import pandas as pd
import numpy as np
from pandas.plotting import register_matplotlib_converters
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


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


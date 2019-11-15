import pandas as pd
import numpy as np
from pandas.plotting import register_matplotlib_converters
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler

def normalization(data: pd.DataFrame):
    # I don't want to normalize the class column
    # So i save it in data_class
    """
    data_class = data['class']
    transf = Normalizer().fit(data) # Normalize it as usual
    data = pd.DataFrame(transf.transform(data, copy=True), columns= data.columns)
    data.pop('class') # Pop the norm_class
    data['class'] = data_class # Add the right class
    """

    # MinMax Normalizer avoids negative values
    data_class = data['class']
    transf = MinMaxScaler().fit(data) # Normalize it as usual
    data = pd.DataFrame(transf.transform(data), columns= data.columns)
    data.pop('class') # Pop the norm_class
    data['class'] = data_class # Add the right class

    return data

def run(data: pd.DataFrame):
    register_matplotlib_converters()

    norm_data = normalization(data)
    return norm_data
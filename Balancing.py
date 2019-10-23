import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import plot_functions as func

from pandas.plotting import register_matplotlib_converters
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from imblearn.over_sampling import SMOTE, RandomOverSampler

'''
Prints the values for the class, and plots a bar_chart showing the
    difference
'''
def unbalanced_data(data):
    target_count = data['class'].value_counts()
    plt.figure()
    plt.title('Class balance')
    plt.bar(target_count.index, target_count.values)

    min_class = target_count.idxmin()
    ind_min_class = target_count.index.get_loc(min_class)

    print('Minority class(', ind_min_class, '):', target_count[ind_min_class]) 
    print('Majority class(', 1-ind_min_class, '):',  target_count[1 - ind_min_class]) 
    print('Proportion:', round(target_count[ind_min_class] / target_count[1-ind_min_class], 2), ': 1')

    plt.show() 
    return 0

'''
Balances the given dataset and a strategy using SMOTE
Shows a bar_chart with the given result
'''
def balance_SMOTE(data, strategy: str, random_state=42):
    target_count = data['class'].value_counts()
    min_class = target_count.idxmin()
    ind_min_class = target_count.index.get_loc(min_class)

    values = {'Original': [target_count.values[ind_min_class], target_count.values[1-ind_min_class]]}

    df_class_min = data[data['class'] == min_class]
    df_class_max = data[data['class'] != min_class]

    df_under = df_class_max.sample(len(df_class_min))
    values['UnderSample'] = [target_count.values[ind_min_class], len(df_under)]

    df_over = df_class_min.sample(len(df_class_max), replace=True)
    values['OverSample'] = [len(df_over), target_count.values[1 - ind_min_class]]

    smote = SMOTE(sampling_strategy=strategy, random_state=random_state)
    y = data.pop('class').values
    X = data.values
    smote_X, smote_y = smote.fit_sample(X, y)
    smote_target_count = pd.Series(smote_y).value_counts()
    values['SMOTE'] = [smote_target_count.values[ind_min_class], smote_target_count.values[1 - ind_min_class]]

    plt.figure()
    func.multiple_bar_chart(plt.gca(), 
                        [target_count.index[ind_min_class], target_count.index[1-ind_min_class]], 
                        values, 'Target', 'frequency', 'Class balance')

    plt.show()

    new_data = pd.concat([pd.DataFrame(smote_X), pd.DataFrame(smote_y)], axis=1)

    return new_data    


def run(data: pd.DataFrame, strategy: str, random_number):
    register_matplotlib_converters()
    
    #This needs to be a copy
    data1 = data.copy(deep=True)
    
    unbalanced_data(data1) #Shows unbalnced data
    new_data = balance_SMOTE(data1, strategy, random_number) # Shows Smote, and returns new_data
    
    new_data.columns = data.columns.tolist()
    return new_data
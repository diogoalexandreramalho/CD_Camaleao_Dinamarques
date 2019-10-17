import pandas as pd
from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt
import plot_functions as func
import re
import json
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split


register_matplotlib_converters()
data = pd.read_csv('pd_speech_features.csv', sep=',', decimal='.', skiprows=1)
data2 = pd.read_csv('pd_speech_features.csv', sep=',', decimal='.')

print(data)


# Creates a dic with a list of columns names associated to the titles given in the csv file
def general_dic(write_file):
    dic = {"Start": []}
    current_title = "Start"

    for title in data2:
        x = re.findall("^Unnamed", title)
        if (x):
            dic[current_title] += [data2[title][0]]
        else:
            dic[title] = [data2[title][0]]
            current_title = title
    
    if write_file:
        with open('dic.txt', 'w') as file:
            file.write(json.dumps(dic))
        
    return dic

# receives a group of columns and divides them according to their initial letters
def group_dic(group_data, key_len, write_file):

    initials = []
    group_dic = {}

    for col in group_data:
        if col[:key_len] not in initials:
            initials += [col[:key_len]]
            group_dic[col[:key_len]] = [col]
        else:
            group_dic[col[:key_len]] += [col]
    
    if write_file:
        with open('dic.txt', 'w') as file:
            file.write(json.dumps(group_dic))
    
    return group_dic

# get the data associated to a set of variables
def get_group_of_data(dic,group_name):
    lst = dic[group_name]
    return data[lst]

# correlate a group of variables
def group_correlation(group_data):
    plt.figure(figsize=[14,7])
    corr_mtx = group_data.corr()
    sns.heatmap(corr_mtx, xticklabels=corr_mtx.columns, yticklabels=corr_mtx.columns, annot=False, cmap='Blues')
    plt.title('Correlation analysis')
    plt.show()


def get_partial_group_of_data(group_data, reg_expression):
    group_lst = []

    for var in group_data:
        x = re.findall(reg_expression, var)
        if (x):
            group_lst += [var]
    
    return data[group_lst]


#Produz todas as variaveis que tem o mesmo sufixo
def produce_allvariables(s,n):
    lst = []
    for i in range(1,n):
        v = s + str(i)
        lst.append(v)
    return lst

#Adiciona uma variavel ao dataset, com o valor correspondete a media das variaveis em var_lst
def add_variable_from_mean(new_var, var_lst):
    ## A usar data global
    data.insert(0,new_var,0)

    for i in range(756):
        num = 0
        for col in var_lst:
            num+= data[col][i]
        
        num/= len(var_lst)
        data[new_var][i] = num



####### DISTRIBUTIONS ############



dic = general_dic(False)
group_data = get_group_of_data(dic,"Wavelet Features")
group_dic = group_dic(group_data, 10, False)


######### TQWT FEATURES ############################

tqwt_features = get_group_of_data(dic,"TQWT Features")

print(data.describe())


######### TQWT GROUPS ################################

tqwt_energy = get_partial_group_of_data(tqwt_features, "^tqwt_energy.*")
tqwt_entropy_shannon = get_partial_group_of_data(tqwt_features, "^tqwt_entropy_shannon.*")
tqwt_entropy_log = get_partial_group_of_data(tqwt_features, "^tqwt_entropy_log.*")
tqwt_TKEO_mean = get_partial_group_of_data(tqwt_features, "^tqwt_TKEO_mean.*")
tqwt_TKEO_std = get_partial_group_of_data(tqwt_features, "^tqwt_TKEO_std.*")
tqwt_medianValue = get_partial_group_of_data(tqwt_features, "^tqwt_medianValue.*")
tqwt_meanValue = get_partial_group_of_data(tqwt_features, "^tqwt_meanValue.*")
tqwt_stdValue = get_partial_group_of_data(tqwt_features, "^tqwt_stdValue.*")
tqwt_minValue = get_partial_group_of_data(tqwt_features, "^tqwt_minValue.*")
tqwt_maxValue = get_partial_group_of_data(tqwt_features, "^tqwt_maxValue.*")
tqwt_skewnessValue = get_partial_group_of_data(tqwt_features, "^tqwt_skewnessValue.*")
tqwt_kurtosisValue = get_partial_group_of_data(tqwt_features, "^tqwt_kurtosisValue.*")



####### HEATMAPS TQWT FEATURES #########################

"""
group_correlation(tqwt_energy)
group_correlation(tqwt_entropy_shannon)
group_correlation(tqwt_entropy_log)
group_correlation(tqwt_TKEO_mean)
group_correlation(tqwt_TKEO_std)
group_correlation(tqwt_medianValue)
group_correlation(tqwt_meanValue)
group_correlation(tqwt_stdValue)
group_correlation(tqwt_minValue)
group_correlation(tqwt_maxValue)
group_correlation(tqwt_skewnessValue)
group_correlation(tqwt_kurtosisValue)

"""


##########  PRODUCE LIST OF ALL VARIABLES FOR A GROUP ########## 


tqwt_energy_lst = produce_allvariables("tqwt_energy_dec_",37)
tqwt_entropy_shannon_lst = produce_allvariables("tqwt_entropy_shannon_dec_",37)
tqwt_entropy_log_lst = produce_allvariables("tqwt_entropy_log_dec_",37)
tqwt_TKEO_mean_lst = produce_allvariables("tqwt_TKEO_mean_dec_",37)
tqwt_TKEO_std_lst = produce_allvariables("tqwt_TKEO_std_dec_",37)
tqwt_medianValue_lst = produce_allvariables("tqwt_medianValue_dec_",37)
tqwt_meanValue_lst = produce_allvariables("tqwt_meanValue_dec_",37)
tqwt_stdValue_lst = produce_allvariables("tqwt_stdValue_dec_",37)
tqwt_minValue_lst = produce_allvariables("tqwt_minValue_dec_",37)
tqwt_maxValue_lst = produce_allvariables("tqwt_maxValue_dec_",37)
tqwt_skewnessValue_lst = produce_allvariables("tqwt_skewnessValue_dec_",37)
tqwt_kurtosisValue_lst = produce_allvariables("tqwt_kurtosisValue_dec_",37)

print(tqwt_entropy_shannon_lst)




######## ADD NEW VARIABLE IN DATASET THAT REPRESENTS A GROUP ######


add_variable_from_mean("tqwt_energy",tqwt_energy_lst)
add_variable_from_mean("tqwt_entropy_shannon",tqwt_entropy_shannon_lst)
add_variable_from_mean("tqwt_entropy_log",tqwt_entropy_log_lst)
add_variable_from_mean("tqwt_TKEO_mean",tqwt_TKEO_mean_lst)
add_variable_from_mean("tqwt_TKEO_std",tqwt_TKEO_std_lst)
add_variable_from_mean("tqwt_medianValue",tqwt_medianValue_lst)
add_variable_from_mean("tqwt_meanValue",tqwt_meanValue_lst)
add_variable_from_mean("tqwt_stdValue",tqwt_energy_lst)
add_variable_from_mean("tqwt_minValue",tqwt_minValue_lst)
add_variable_from_mean("tqwt_maxValue",tqwt_maxValue_lst)
add_variable_from_mean("tqwt_skewnessValue",tqwt_skewnessValue_lst)
add_variable_from_mean("tqwt_kurtosisValue",tqwt_kurtosisValue_lst)


print(data)


###########  CLASSIFICATION ###########################

"""

data = tqwt_energy
print(data.columns)

y: np.ndarray = data.pop('tqwt_energy_dec_1').values
X: np.ndarray = data.values
labels: np.ndarray = pd.unique(y)

trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)


clf = GaussianNB()
clf.fit(trnX, trnY)
prdY = clf.predict(tstX)
cnf_mtx = metrics.confusion_matrix(tstY, prdY, labels)
func.plot_confusion_matrix(plt.gca(), cnf_mtx, labels)

"""

"""

group_correlation(tqwt_energy)

group_correlation(tqwt_entropy)

group_correlation(tqwt_TKEO)



group_correlation(tqwt_medianValue)
group_correlation(tqwt_meanValue)




for group in group_dic:
    group_correlation(group_data[group_dic[group]])
    
#mfcc_std = get_partial_group_of_data(mfcc,"^std_.*")
#group_correlation(mfcc_std)

#mfcc_mean_delta = get_partial_group_of_data(group_data,"^E")
#print(mfcc_mean_delta)
#group_correlation(mfcc_mean_delta)
mfcc_std = get_partial_group_of_data(mfcc,"^std_.*")
group_correlation(dic,mfcc_std)



mfcc_mean = get_partial_group_of_data(mfcc,"^mean_.*")
group_correlation(dic,mfcc_mean)


base_features = get_group_of_data(dic,"Baseline Features")

shimmer = get_partial_group_of_data(base_features, "^.*Shimmer")
group_correlation(dic,shimmer)

pulses = get_partial_group_of_data(base_features, "^.*Pulses")
group_correlation(dic,pulses)

columns = tqwt_energy.columns
rows, cols = func.choose_grid(len(columns))
plt.figure()
fig, axs = plt.subplots(rows, cols, figsize=(cols*4, rows*4), squeeze=False)
i, j = 0, 0
for n in range(len(columns)):
    axs[i, j].set_title('Boxplot for %s'%columns[n])
    axs[i, j].boxplot(data[columns[n]].dropna().values)
    i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
fig.tight_layout()
plt.show()
"""


#group_correlation(dic,group_data)
#boxplot(get_partial_group_of_data(group_data, "Jitter$"),0,0.02)
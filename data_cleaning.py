import pandas as pd
from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt
import plot_functions as func
import re
import json
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE, RandomOverSampler


register_matplotlib_converters()
data = pd.read_csv('pd_speech_features.csv', sep=',', decimal='.', skiprows=1)



# Creates a dic with a list of columns names associated to the titles given in the csv file
def general_dic(write_file):
	data2 = pd.read_csv('pd_speech_features.csv', sep=',', decimal='.')
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
def get_data_from_dic(dic,group_name):
	lst = dic[group_name]
	return data[lst]


# get group of data associated based on a regular expression
def get_data_by_expression(group_data, reg_expression):
	group_lst = []

	for var in group_data:
		x = re.findall(reg_expression, var)
		if (x):
			group_lst += [var]
	
	return data[group_lst]


# correlate a group of variables
def group_correlation(group_data):
	plt.figure(figsize=[14,7])
	corr_mtx = group_data.corr()
	sns.heatmap(corr_mtx, xticklabels=corr_mtx.columns, yticklabels=corr_mtx.columns, annot=False, cmap='Blues')
	plt.title('Correlation analysis')
	plt.show()


# calculates new variable from the mean of a group of variables
def add_variable_from_mean(data, new_var, vars_lst, delete):
	data.insert(0, new_var, 0)

	for i in range(756):
		num = 0
		for col in vars_lst:
			num += data[col][i]
		
		num /= len(vars_lst)
		
		data[new_var][i] = num
	
	if delete:
		for var in vars_lst:
			del data[var]
	
	
	return data


	

# deletes given columns from data set
def delete_columns(lst,data):
	for var in lst:
		del data[var]
	return data


#Produz todas as variaveis que tem o mesmo sufixo
def produce_allvariables(s,n):
	lst = []
	for i in range(1,n):
		v = s + str(i)
		lst.append(v)
	return lst




##############################
####### DISTRIBUTIONS ########
##############################



def bandwidth_parameters(dic):
	bp_data = get_data_from_dic(dic,"Bandwidth Parameters")
	#group_correlation(bp_data)
	return bp_data



def baseline_features(dic):
	bf_data = get_data_from_dic(dic,"Baseline Features")

	shimmer = get_data_by_expression(bf_data, "^.*Shimmer")
	group_correlation(shimmer)

	pulses = get_data_by_expression(bf_data, "^.*Pulses")
	group_correlation(pulses)



def formant_frequencies(dic):
	ff_data = get_data_from_dic(dic,"Formant Frequencies")
	#group_correlation(ff_data)
	return ff_data



def intensity_parameters(dic, correlation):
	ip_data = get_data_from_dic(dic,"Intensity Parameters")
	if correlation:
		group_correlation(ip_data)
	else:
		return ip_data['meanIntensity']





def mfcc(dic):
	mfcc_data = get_data_from_dic(dic,"MFCC ")

	# std
	std = get_data_by_expression(mfcc_data,"^std_.*")
	group_correlation(std)

	# mean
	mean = get_data_by_expression(mfcc_data,"^mean_.*")
	group_correlation(mean)
	
	# mean_MFCC
	mean_MFCC = get_data_by_expression(mfcc_data,"^mean_MFCC_.*")
	mean_MFCC = pd.concat([mean_MFCC, mfcc_data['mean_Log_energy']], axis=1, sort=False)
	mean_MFCC = add_variable_from_mean(mean_MFCC, 'mean_MFCC_master', list(mean_MFCC.columns), False)
	#group_correlation(mean_MFCC)
	
	# mean_deltas, mean_master is equal to 0!!
	mean_delta = get_data_by_expression(mfcc_data,"^mean_.*delta$")
	mean_delta = pd.concat([mean_delta, mfcc_data['mean_delta_log_energy'], mfcc_data['mean_delta_delta_log_energy']], axis=1, sort=False)
	mean_delta = add_variable_from_mean(mean_delta, 'mean_delta_master', list(mean_delta.columns), False)
	#group_correlation(mean_delta)
	
	# std_MFCC
	std_MFCC = get_data_by_expression(mfcc_data,"^std_MFCC_.*")
	std_MFCC = pd.concat([std_MFCC, mfcc_data['std_Log_energy']], axis=1, sort=False)
	std_MFCC = add_variable_from_mean(std_MFCC, 'std_MFCC_master', list(std_MFCC.columns), False)
	#group_correlation(std_MFCC)
	
	# std_deltas
	std_delta = get_data_by_expression(mfcc_data,"^std_.*delta$")
	std_delta = pd.concat([std_delta, mfcc_data['std_delta_log_energy'], mfcc_data['std_delta_delta_log_energy']], axis=1, sort=False)
	std_delta = add_variable_from_mean(std_delta, 'std_delta_master', list(std_delta.columns), False)
	#group_correlation(std_delta)
	
	# std_MFCC and mean_MFCC
	mean_std_MFCC = pd.concat([mean_MFCC, std_MFCC], axis=1, sort=False)
	group_correlation(mean_std_MFCC)
	
	# std_deltas and mean_deltas
	mean_std_deltas = pd.concat([mean_delta, std_delta], axis=1, sort=False)
	group_correlation(mean_std_deltas)




def vocal_fold(dic):
	vf_data = get_data_from_dic(dic,"Vocal Fold")




def wavelet_features(dic, write_pickle):
	wf_data = get_data_from_dic(dic,"Wavelet Features")
	dic_by_len = group_dic(wf_data, 13, False)
	
	###### det_entropy ######
	"""
	det_entropy_data = wf_data[dic_by_len['det_entropy_s']]

	det_entropy_1_to_3_lst = ['det_entropy_shannon_1_coef', 'det_entropy_shannon_2_coef', 'det_entropy_shannon_3_coef']
	det_entropy_data = add_variable_from_mean(det_entropy_data, 'det_entropy_1_to_3', det_entropy_1_to_3_lst, 1)

	det_entropy_8_and_10_lst = ['det_entropy_shannon_8_coef', 'det_entropy_shannon_10_coef']
	det_entropy_data = add_variable_from_mean(det_entropy_data, 'det_entropy_8_and_10', det_entropy_8_and_10_lst, 1)
	
	group_correlation(det_entropy_data)
	"""

	###### det_TKEO_mean ######

	if write_pickle:
		det_TKEO_m_data = wf_data[dic_by_len['det_TKEO_mean']]
		det_TKEO_m_1_to_3_lst = ['det_TKEO_mean_1_coef', 'det_TKEO_mean_2_coef', 'det_TKEO_mean_3_coef']
		det_TKEO_m_data = add_variable_from_mean(det_TKEO_m_data, 'det_TKEO_m_1_to_3', det_TKEO_m_1_to_3_lst, 1)

		det_TKEO_m_8_to_10_lst = ['det_TKEO_mean_8_coef', 'det_TKEO_mean_9_coef', 'det_TKEO_mean_10_coef']
		det_TKEO_m_data = add_variable_from_mean(det_TKEO_m_data, 'det_TKEO_m_8_to_10', det_TKEO_m_8_to_10_lst, 1)

		det_TKEO_m_6_and_7_lst = ['det_TKEO_mean_6_coef', 'det_TKEO_mean_7_coef']
		det_TKEO_m_data = add_variable_from_mean(det_TKEO_m_data, 'det_TKEO_m_6_and_7', det_TKEO_m_6_and_7_lst, 1)

		det_TKEO_m_data.to_pickle("det_TKEO_m_data.pkl")

		#group_correlation(det_TKEO_m_data)
	else:
		det_TKEO_m_data = pd.read_pickle("det_TKEO_m_data.pkl")

	return det_TKEO_m_data



def tqwt_features(dic, data):

	tqwt_data = get_data_from_dic(dic,"TQWT Features")
	#print(data.describe())

	
	# groups of data
	tqwt_energy = get_data_by_expression(tqwt_data, "^tqwt_energy.*")
	tqwt_entropy_shannon = get_data_by_expression(tqwt_data, "^tqwt_entropy_shannon.*")
	tqwt_entropy_log = get_data_by_expression(tqwt_data, "^tqwt_entropy_log.*")
	tqwt_TKEO_mean = get_data_by_expression(tqwt_data, "^tqwt_TKEO_mean.*")
	tqwt_TKEO_std = get_data_by_expression(tqwt_data, "^tqwt_TKEO_std.*")
	tqwt_medianValue = get_data_by_expression(tqwt_data, "^tqwt_medianValue.*")
	tqwt_meanValue = get_data_by_expression(tqwt_data, "^tqwt_meanValue.*")
	tqwt_stdValue = get_data_by_expression(tqwt_data, "^tqwt_stdValue.*")
	tqwt_minValue = get_data_by_expression(tqwt_data, "^tqwt_minValue.*")
	tqwt_maxValue = get_data_by_expression(tqwt_data, "^tqwt_maxValue.*")
	tqwt_skewnessValue = get_data_by_expression(tqwt_data, "^tqwt_skewnessValue.*")
	tqwt_kurtosisValue = get_data_by_expression(tqwt_data, "^tqwt_kurtosisValue.*")
	

	# produce list of variables for a group  
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


	"""
	# add new variable in a dataset that represents a group ######
	data = add_variable_from_mean(data, "tqwt_energy",tqwt_energy_lst, 1)
	data = add_variable_from_mean(data, "tqwt_entropy_shannon",tqwt_entropy_shannon_lst, 1)
	data = add_variable_from_mean(data, "tqwt_entropy_log",tqwt_entropy_log_lst, 1)
	data = add_variable_from_mean(data, "tqwt_TKEO_mean",tqwt_TKEO_mean_lst, 1)
	data = add_variable_from_mean(data, "tqwt_TKEO_std",tqwt_TKEO_std_lst, 1)
	data = add_variable_from_mean(data, "tqwt_medianValue",tqwt_medianValue_lst, 1)
	data = add_variable_from_mean(data, "tqwt_meanValue",tqwt_meanValue_lst, 1)
	data = add_variable_from_mean(data, "tqwt_stdValue",tqwt_stdValue_lst, 1)
	data = add_variable_from_mean(data, "tqwt_minValue",tqwt_minValue_lst, 1)
	data = add_variable_from_mean(data, "tqwt_maxValue",tqwt_maxValue_lst, 1)
	data = add_variable_from_mean(data, "tqwt_skewnessValue",tqwt_skewnessValue_lst, 1)
	data = add_variable_from_mean(data, "tqwt_kurtosisValue",tqwt_kurtosisValue_lst, 1)
	"""

	# add new variable in a dataset that represents a group ######
	print("1/12")
	tqwt_energy = add_variable_from_mean(tqwt_energy, "tqwt_energy",tqwt_energy_lst, 0)
	print("2/12")
	tqwt_entropy_shannon = add_variable_from_mean(tqwt_entropy_shannon, "tqwt_entropy_shannon",tqwt_entropy_shannon_lst, 0)
	print("3/12")
	tqwt_entropy_log = add_variable_from_mean(tqwt_entropy_log, "tqwt_entropy_log",tqwt_entropy_log_lst, 0)
	print("4/12")
	tqwt_TKEO_mean = add_variable_from_mean(tqwt_TKEO_mean, "tqwt_TKEO_mean",tqwt_TKEO_mean_lst, 0)
	print("5/12")
	tqwt_TKEO_std = add_variable_from_mean(tqwt_TKEO_std, "tqwt_TKEO_std",tqwt_TKEO_std_lst, 0)
	print("6/12")
	tqwt_medianValue = add_variable_from_mean(tqwt_medianValue, "tqwt_medianValue",tqwt_medianValue_lst, 0)
	print("7/12")
	tqwt_meanValue = add_variable_from_mean(tqwt_meanValue, "tqwt_meanValue",tqwt_meanValue_lst, 0)
	print("8/12")
	tqwt_stdValue = add_variable_from_mean(tqwt_stdValue, "tqwt_stdValue",tqwt_stdValue_lst, 0)
	print("9/12")
	tqwt_minValue = add_variable_from_mean(tqwt_minValue, "tqwt_minValue",tqwt_minValue_lst, 0)
	print("10/12")
	tqwt_maxValue = add_variable_from_mean(tqwt_maxValue, "tqwt_maxValue",tqwt_maxValue_lst, 0)
	print("11/12")
	tqwt_skewnessValue = add_variable_from_mean(tqwt_skewnessValue, "tqwt_skewnessValue",tqwt_skewnessValue_lst, 0)
	print("12/12")
	tqwt_kurtosisValue = add_variable_from_mean(tqwt_kurtosisValue, "tqwt_kurtosisValue",tqwt_kurtosisValue_lst, 0)



	# heatmaps for different groups
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


	#return data


dic = general_dic(False)


#mfcc(dic)
#ff_data = formant_frequencies(dic)
#bp_data = bandwidth_parameters(dic)
#ip_data = intensity_parameters(dic, 0)

#new_data = pd.concat([ff_data, bp_data, ip_data, data['class']], axis=1, sort=False)
#print(new_data)
#tqwt_features(dic, data)






target_count = data['class'].value_counts()

plt.figure()
plt.title('Class balance')
plt.bar(target_count.index, target_count.values)
plt.show()

min_class = target_count.idxmin()
ind_min_class = target_count.index.get_loc(min_class)

print('Minority class:', target_count[ind_min_class])
print('Majority class:', target_count[1-ind_min_class])
print('Proportion:', round(target_count[ind_min_class] / target_count[1-ind_min_class], 2), ': 1')


##### SMOTE #####

RANDOM_STATE = 42
values = {'Original': [target_count.values[ind_min_class], target_count.values[1-ind_min_class]]}

df_class_min = data[data['class'] == min_class]
df_class_max = data[data['class'] != min_class] 

df_under = df_class_max.sample(len(df_class_min))
values['UnderSample'] = [target_count.values[ind_min_class], len(df_under)]

df_over = df_class_min.sample(len(df_class_max), replace=True)
values['OverSample'] = [len(df_over), target_count.values[1-ind_min_class]]

smote = SMOTE(ratio='minority', random_state=RANDOM_STATE)

y = data.pop('class').values
print(data)

X = data.values
_, smote_y = smote.fit_sample(X, y)
smote_target_count = pd.Series(smote_y).value_counts()
values['SMOTE'] = [smote_target_count.values[ind_min_class], smote_target_count.values[1-ind_min_class]]

plt.figure()
func.multiple_bar_chart(plt.gca(), 
                        [target_count.index[ind_min_class], target_count.index[1-ind_min_class]], 
                        values, 'Target', 'frequency', 'Class balance')
plt.show()











#########################################
############  NORMALIZATION  ############
#########################################


#transf = MinMaxScaler(data, copy=True)
#data = pd.DataFrame(transf.transform(data), columns= data.columns)


scaler = MinMaxScaler()
scaler.fit(data)
MinMaxScaler(copy=True,feature_range=(0,1))
norm_data = pd.DataFrame(scaler.transform(data), columns = data.columns)
#print(norm_data.keys())



#######################################
###########  CLASSIFICATION ###########
#######################################


y: np.ndarray = norm_data.pop('class').values
X: np.ndarray = norm_data.values
labels = pd.unique(y)


trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)



def knn():

	##### BEST N = 5 #######

	nvalues = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41]
	dist = ['manhattan', 'euclidean', 'chebyshev']
	values = {}
	for d in dist:
		yvalues = []
		for n in nvalues:
			knn = KNeighborsClassifier(n_neighbors=n, metric=d)
			knn.fit(trnX, trnY)
			prdY = knn.predict(tstX)
			yvalues.append(metrics.accuracy_score(tstY, prdY))
		values[d] = yvalues

	plt.figure()
	func.multiple_line_chart(plt.gca(), nvalues, values, 'KNN variants', 'n', 'accuracy', percentage=True)
	plt.show()

def naive_bayes():

	clf = GaussianNB()
	clf.fit(trnX, trnY)
	prdY = clf.predict(tstX)
	cnf_mtx = metrics.confusion_matrix(tstY, prdY, labels)
	
	estimators = {'GaussianNB': GaussianNB(), 
			  'MultinomialNB': MultinomialNB(), 
			  'BernoulyNB': BernoulliNB()}

	xvalues = []
	yvalues = []
	for clf in estimators:
		xvalues.append(clf)
		estimators[clf].fit(trnX, trnY)
		prdY = estimators[clf].predict(tstX)
		yvalues.append(metrics.accuracy_score(tstY, prdY))

	plt.figure()
	func.bar_chart(plt.gca(), xvalues, yvalues, 'Comparison of Naive Bayes Models', '', 'accuracy', percentage=True)
	plt.show()




#knn()
#naive_bayes()



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
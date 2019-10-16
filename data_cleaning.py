import pandas as pd
from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt
import plot_functions as func
import re
import json
import seaborn as sns


register_matplotlib_converters()
data = pd.read_csv('pd_speech_features.csv', sep=',', decimal='.', skiprows=1)
data2 = pd.read_csv('pd_speech_features.csv', sep=',', decimal='.')


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
def get_data_from_dic(dic,group_name):
	lst = dic[group_name]
	return data[lst]

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
	sns.heatmap(corr_mtx, xticklabels=corr_mtx.columns, yticklabels=corr_mtx.columns, annot=True, cmap='Blues')
	plt.title('Correlation analysis')
	plt.show()


def add_variable_from_mean(data, new_var, vars_lst):
	data.insert(0, new_var, 0)

	for i in range(756):
		num = 0
		for col in vars_lst:
			num += data[col][i]
		
		num /= len(vars_lst)
		data[new_var][i] = num
		print(i)
	
	return data
	

def delete_columns(lst,data):
	for var in lst:
		del data[var]
	return data



####### DISTRIBUTIONS ############



##############################
#### Bandwidth Parameters ####
##############################

def bandwidth_parameters():
	bp_data = get_data_from_dic(dic,"Bandwidth Parameters")
	group_correlation(bp_data)


###########################
#### Baseline Features ####
###########################

def baseline_features():
	bf_data = get_data_from_dic(dic,"Baseline Features")

	shimmer = get_data_by_expression(bf_data, "^.*Shimmer")
	group_correlation(shimmer)

	pulses = get_data_by_expression(bf_data, "^.*Pulses")
	group_correlation(pulses)



#############################
#### Formant Frequencies ####
#############################

def formant_frequencies():
	ff_data = get_data_from_dic(dic,"Baseline Features")
	group_correlation(ff_data)


##############################
#### Intensity Parameters ####
##############################

def intensity_parameters():
	ip_data = get_data_from_dic(dic,"Intensity Parameters")
	group_correlation(ip_data)

##############
#### MFCC ####
##############

def mfcc():
	mfcc_data = get_data_from_dic(dic,"MFCC ")
	
	mfcc_std = get_data_by_expression(mfcc_data,"^std_.*")
	group_correlation(mfcc_std)

	mfcc_mean = get_data_by_expression(mfcc_data,"^mean_.*")
	group_correlation(mfcc_mean)

	mfcc_E = get_data_by_expression(mfcc_data,"^E")
	group_correlation(mfcc_E)


####################
#### Vocal Fold ####
####################

def vocal_fold():
	vf_data = get_data_from_dic(dic,"Vocal Fold")


##########################
#### Wavelet Features ####
##########################


def wavelet_features():
	wf_data = get_data_from_dic(dic,"Wavelet Features")
	dic_by_len = group_dic(wf_data, 13, False)
	
	###### det_entropy ######
	det_entropy_data = wf_data[dic_by_len['det_entropy_s']]

	det_entropy_1_to_3_lst = ['det_entropy_shannon_1_coef', 'det_entropy_shannon_2_coef', 'det_entropy_shannon_3_coef']
	det_entropy_data = add_variable_from_mean(det_entropy_data, 'det_entropy_1_to_3', det_entropy_1_to_3_lst)
	delete_columns(det_entropy_1_to_3_lst, det_entropy_data)

	det_entropy_8_and_10_lst = ['det_entropy_shannon_8_coef', 'det_entropy_shannon_10_coef']
	det_entropy_data = add_variable_from_mean(det_entropy_data, 'det_entropy_8_and_10', det_entropy_8_and_10_lst)
	delete_columns(det_entropy_8_and_10_lst, det_entropy_data)


	group_correlation(det_entropy_data)


dic = general_dic(False)
wavelet_features()

import pandas as pd
from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt
import plot_functions as func
import re
import json
import seaborn as sns
import numpy as np


register_matplotlib_converters()
data = pd.read_csv('Data/pd_speech_features.csv', sep=',', decimal='.', skiprows=1)



# Creates a dic with a list of columns names associated to the titles given in the csv file
def general_dic(write_file):
	data2 = pd.read_csv('Data/pd_speech_features.csv', sep=',', decimal='.')
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
	sns.heatmap(corr_mtx, xticklabels=corr_mtx.columns, yticklabels=corr_mtx.columns, annot=True, cmap='Blues')
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
		
		data.loc[i,new_var] = num
	

	if delete:
		for var in vars_lst:
			del data[var]
	
	
	return data


def delete_columns(data, vars_lst):
	for var in vars_lst:
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

dic = general_dic(False)


def baseline_features(dic, ratio, correlations, pickles, write):
	if not write:
		if ratio == 0.97:
			new_bf_data = pd.read_pickle("Pickles/Baseline/baseline_97.pkl")
		elif ratio == 0.95:
			new_bf_data = pd.read_pickle("Pickles/Baseline/baseline_95.pkl")
		elif ratio == 0.9:
			new_bf_data = pd.read_pickle("Pickles/Baseline/baseline_90.pkl")
		elif ratio == 0.85:
			new_bf_data = pd.read_pickle("Pickles/Baseline/baseline_85.pkl")
		elif ratio == 0.8:
			new_bf_data = pd.read_pickle("Pickles/Baseline/baseline_80.pkl")

	else:
		bf_data = get_data_from_dic(dic,"Baseline Features")

		# Shimmer
		if ratio > 0.89:
			if pickles[0]:
				shimmer = get_data_by_expression(bf_data, "^.*Shimmer")
				lst = list(shimmer.columns)
				lst.remove('apq11Shimmer')
				shimmer = add_variable_from_mean(shimmer, 'shimmer_master', lst, True)
				shimmer.to_pickle("Pickles/Baseline/shimmer_+89.pkl")
			else:
				shimmer = pd.read_pickle("Pickles/Baseline/shimmer_+89.pkl")
		else:
			if pickles[0]:
				shimmer = get_data_by_expression(bf_data, "^.*Shimmer")
				shimmer = add_variable_from_mean(shimmer, 'shimmer_master', list(shimmer.columns), True)
				shimmer.to_pickle("Pickles/Baseline/shimmer_-89.pkl")
			else:
				shimmer = pd.read_pickle("Pickles/Baseline/shimmer_-89.pkl")


		# Jitter
		if pickles[1]:
			jitter = get_data_by_expression(bf_data, "^.*Jitter")
			jitter = add_variable_from_mean(jitter, 'jitter_master', list(jitter.columns), True)
			jitter.to_pickle("Pickles/Baseline/jitter.pkl")
		else:
			jitter = pd.read_pickle("Pickles/Baseline/jitter.pkl")


		# Harmonicity
		if ratio > 0.82:
			harmonicity = get_data_by_expression(bf_data, "Harmonicity$")
		else:
			harmonicity = get_data_by_expression(bf_data, "Harmonicity$")
			del harmonicity['meanHarmToNoiseHarmonicity']

		# Pulses
		pulses = get_data_by_expression(bf_data, "^.*Pulses")
		del pulses['numPeriodsPulses']


		new_bf_data = data[['PPE','DFA', 'RPDE']]
		new_bf_data = pd.concat([new_bf_data, pulses, jitter, shimmer, harmonicity], axis=1, sort=False)

		if ratio == 0.97:
			new_bf_data.to_pickle("Pickles/Baseline/baseline_97.pkl")
		elif ratio == 0.95:
			new_bf_data.to_pickle("Pickles/Baseline/baseline_95.pkl")
		elif ratio == 0.9:
			new_bf_data.to_pickle("Pickles/Baseline/baseline_90.pkl")
		elif ratio == 0.85:
			new_bf_data.to_pickle("Pickles/Baseline/baseline_85.pkl")
		elif ratio == 0.8:
			new_bf_data.to_pickle("Pickles/Baseline/baseline_80.pkl")
		
		if correlations:
			new_bf_data = get_data_from_dic(dic,"Baseline Features")

			shimmer = get_data_by_expression(new_bf_data, "Shimmer$")
			group_correlation(shimmer)

			jitter = get_data_by_expression(new_bf_data, "Jitter$")
			group_correlation(jitter)

			harmonicity = get_data_by_expression(new_bf_data, "Harmonicity$")
			group_correlation(harmonicity)

			pulses = get_data_by_expression(new_bf_data, "Pulses$")
			group_correlation(pulses)
	
	return new_bf_data


def bandwidth_parameters(dic, correlation, gender_data):
	bp_data = get_data_from_dic(dic,"Bandwidth Parameters")
	if correlation:
		group_correlation(bp_data)
	return bp_data



def formant_frequencies(dic, correlation):
	ff_data = get_data_from_dic(dic,"Formant Frequencies")
	if correlation:
		group_correlation(ff_data)
	return ff_data



def intensity_parameters(dic, ratio, correlations, write):
	if not write:
		if ratio == 0.97 or ratio == 0.95:
			ip_data = pd.read_pickle("Pickles/Intensity/intensity_+91.pkl")
		elif ratio == 0.9 or ratio == 0.85 or ratio == 0.8:
			ip_data = pd.read_pickle("Pickles/Intensity/intensity_-91.pkl")
	else:
		if not correlations:
			ip_data = get_data_from_dic(dic,"Intensity Parameters")

			if ratio > 0.91:
				del ip_data['maxIntensity']
			else:
				ip_data = delete_columns(ip_data, ['maxIntensity', 'minIntensity'])
			
			if ratio == 0.97 or ratio == 0.95:
				ip_data.to_pickle("Pickles/Intensity/intensity_+91.pkl")
			elif ratio == 0.9 or ratio == 0.85 or ratio == 0.8:
				ip_data.to_pickle("Pickles/Intensity/intensity_-91.pkl")

		else:
			ip_data = get_data_from_dic(dic,"Intensity Parameters")
			group_correlation(ip_data)

	return ip_data
		


def mfcc(dic, ratio, correlations, pickles, write):
	mfcc_data = get_data_from_dic(dic,"MFCC ")
	
	if not write:
		if ratio == 0.97:
			new_mfcc_data = pd.read_pickle("Pickles/MFCC/mfcc_97.pkl")
		elif ratio == 0.95:
			new_mfcc_data = pd.read_pickle("Pickles/MFCC/mfcc_95.pkl")
		elif ratio == 0.9:
			new_mfcc_data = pd.read_pickle("Pickles/MFCC/mfcc_90.pkl")
		elif ratio == 0.85:
			new_mfcc_data = pd.read_pickle("Pickles/MFCC/mfcc_85.pkl")
		elif ratio == 0.8:
			new_mfcc_data = pd.read_pickle("Pickles/MFCC/mfcc_80.pkl")
	else:
		
		# mean_coef
		if pickles[0]:
			mean_coef = get_data_by_expression(mfcc_data,"^mean_MFCC_.*")
			mean_coef = pd.concat([mfcc_data['mean_Log_energy'], mean_coef], axis=1, sort=False)
			mean_coef.to_pickle("Pickles/MFCC/mean_coef.pkl")
		else:
			mean_coef = pd.read_pickle("Pickles/MFCC/mean_coef.pkl")
		

		# mean_deltas
		if ratio <= 0.86:
			if pickles[1]:
				mean_delta = get_data_by_expression(mfcc_data,"^mean_(...|....)_delta$")
				mean_delta_delta = get_data_by_expression(mfcc_data,"^mean_(...|....)_delta_delta$")
				mean_deltas = pd.concat([mfcc_data['mean_delta_log_energy'], mean_delta, mfcc_data['mean_delta_delta_log_energy'], mean_delta_delta], axis=1, sort=False)
				lst = ['mean_delta_log_energy', 'mean_0th_delta']
				mean_deltas = add_variable_from_mean(mean_deltas, 'mean_delta_master', lst, True)
				mean_deltas.to_pickle("Pickles/MFCC/mean_deltas_-86.pkl")
			else:
				mean_deltas = pd.read_pickle("Pickles/MFCC/mean_deltas_-86.pkl")
		else:
			if pickles[1]:
				mean_delta = get_data_by_expression(mfcc_data,"^mean_(...|....)_delta$")
				mean_delta_delta = get_data_by_expression(mfcc_data,"^mean_(...|....)_delta_delta$")
				mean_deltas = pd.concat([mfcc_data['mean_delta_log_energy'], mean_delta, mfcc_data['mean_delta_delta_log_energy'], mean_delta_delta], axis=1, sort=False)
				mean_deltas.to_pickle("Pickles/MFCC/mean_deltas_+86.pkl")
			else:
				mean_deltas = pd.read_pickle("Pickles/MFCC/mean_deltas_+86.pkl")

		
		# std_coef
		if pickles[2]:
			std_coef = get_data_by_expression(mfcc_data,"^std_MFCC_.*")
			std_coef = pd.concat([mfcc_data['std_Log_energy'], std_coef], axis=1, sort=False)
			if ratio <= 0.88:
				lst = ['std_Log_energy', 'std_MFCC_0th_coef']
				std_coef = add_variable_from_mean(std_coef, 'std_MFCC_master', lst, True)
				std_coef.to_pickle("Pickles/MFCC/std_coef_-88.pkl")
			else:
				std_coef.to_pickle("Pickles/MFCC/std_coef_+88.pkl")
		else:
			if ratio <= 0.88:
				std_coef = pd.read_pickle("Pickles/MFCC/std_coef_-88.pkl")
			else:
				std_coef = pd.read_pickle("Pickles/MFCC/std_coef_+88.pkl")
		
			
		# std_deltas
		if pickles[3]:
			std_delta = get_data_by_expression(mfcc_data,"^std_(...|....)_delta$")
			std_deltas = pd.concat([mfcc_data['std_delta_log_energy'], std_delta, mfcc_data['std_delta_delta_log_energy']], axis=1, sort=False)
			
			if ratio > 0.95:
				std_delta_delta = get_data_by_expression(mfcc_data,"^std_(...|....)_delta_delta$")
				std_deltas = pd.concat([std_deltas, std_delta_delta], axis=1, sort=False)
				std_deltas.to_pickle("Pickles/MFCC/std_deltas_+95.pkl")
			if ratio <= 0.95:
				lst = ['std_6th_delta_delta', 'std_8th_delta_delta', 'std_11th_delta_delta']
				std_deltas = pd.concat([std_deltas, mfcc_data[lst]], axis=1, sort=False)
				std_deltas.to_pickle("Pickles/MFCC/std_deltas_=95.pkl")
			if ratio <= 0.94:
				lst = ['std_6th_delta_delta', 'std_8th_delta_delta']
				std_deltas = delete_columns(std_deltas, lst)
				std_deltas.to_pickle("Pickles/MFCC/std_deltas_=94.pkl")
			if ratio <= 0.93:
				lst = ['std_11th_delta_delta']
				std_deltas = delete_columns(std_deltas, lst)
				std_deltas.to_pickle("Pickles/MFCC/std_deltas_=93.pkl")
			if ratio <= 0.92:
				lst = ['std_delta_delta_log_energy']
				std_deltas = delete_columns(std_deltas, lst)
				std_deltas.to_pickle("Pickles/MFCC/std_deltas_-92.pkl")
	
		else:
			if ratio > 0.95:
				std_deltas = pd.read_pickle("Pickles/MFCC/std_deltas_+95.pkl")
			if ratio == 0.95:
				std_deltas = pd.read_pickle("Pickles/MFCC/std_deltas_=95.pkl")
			if ratio == 0.94:
				std_deltas = pd.read_pickle("Pickles/MFCC/std_deltas_=94.pkl")
			if ratio == 0.93:
				std_deltas = pd.read_pickle("Pickles/MFCC/std_deltas_=93.pkl")
			if ratio <= 0.92:
				std_deltas = pd.read_pickle("Pickles/MFCC/std_deltas_-92.pkl")
		

	new_mfcc_data = pd.concat([mean_coef, mean_deltas, std_coef, std_deltas], axis=1, sort=False)

	if ratio == 0.97:
		new_mfcc_data.to_pickle("Pickles/MFCC/mfcc_97.pkl")
	elif ratio == 0.95:
		new_mfcc_data.to_pickle("Pickles/MFCC/mfcc_95.pkl")
	elif ratio == 0.9:
		new_mfcc_data.to_pickle("Pickles/MFCC/mfcc_90.pkl")
	elif ratio == 0.85:
		new_mfcc_data.to_pickle("Pickles/MFCC/mfcc_85.pkl")
	elif ratio == 0.8:
		new_mfcc_data.to_pickle("Pickles/MFCC/mfcc_80.pkl")

		
	if correlations:
		
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
		group_correlation(mean_MFCC)
		
		# mean_deltas
		mean_delta = get_data_by_expression(mfcc_data,"^mean_.*delta$")
		mean_delta = pd.concat([mean_delta, mfcc_data['mean_delta_log_energy'], mfcc_data['mean_delta_delta_log_energy']], axis=1, sort=False)
		lst = ['mean_delta_log_energy', 'mean_0th_delta']
		mean_delta = add_variable_from_mean(mean_delta, 'mean_delta_master', lst, False)
		group_correlation(mean_delta)
		
		# std_MFCC
		std_MFCC = get_data_by_expression(mfcc_data,"^std_MFCC_.*")
		std_MFCC = pd.concat([std_MFCC, mfcc_data['std_Log_energy']], axis=1, sort=False)
		lst = ['std_Log_energy', 'std_MFCC_0th_coef']
		std_MFCC = add_variable_from_mean(std_MFCC, 'std_MFCC_master', lst, False)
		group_correlation(std_MFCC)
		
		# std_deltas
		std_delta = get_data_by_expression(mfcc_data,"^std_.*delta$")
		std_delta = pd.concat([std_delta, mfcc_data['std_delta_log_energy'], mfcc_data['std_delta_delta_log_energy']], axis=1, sort=False)
		std_delta = add_variable_from_mean(std_delta, 'std_delta_master', list(std_delta.columns), False)
		group_correlation(std_delta)
		
		# std_MFCC and mean_MFCC
		mean_std_MFCC = pd.concat([mean_MFCC, std_MFCC], axis=1, sort=False)
		group_correlation(mean_std_MFCC)
		
		# std_deltas and mean_deltas
		mean_std_deltas = pd.concat([mean_delta, std_delta], axis=1, sort=False)
		group_correlation(mean_std_deltas)
		

	return new_mfcc_data



def vocal_fold(dic, ratio, correlations, pickles, write):
	vf_data = get_data_from_dic(dic,"Vocal Fold")

	if not write:
		if ratio == 0.97:
			new_vf_data = pd.read_pickle("Pickles/Vocal/vocal_97.pkl")
		elif ratio == 0.95:
			new_vf_data = pd.read_pickle("Pickles/Vocal/vocal_97.pkl")
		elif ratio == 0.9:
			new_vf_data = pd.read_pickle("Pickles/Vocal/vocal_97.pkl")
		elif ratio == 0.85:
			new_vf_data = pd.read_pickle("Pickles/Vocal/vocal_97.pkl")
		elif ratio == 0.8:
			new_vf_data = pd.read_pickle("Pickles/Vocal/vocal_97.pkl")
	
	else:

		"""
		# GQ
		if pickles[0]:
			gq = get_data_by_expression(vf_data,"^GQ")
			gq.to_pickle("Pickles/Vocal/GQ.pkl")
		else:
			gq = pd.read_pickle("Pickles/Vocal/GQ.pkl")
		
		
		# GNE
		if pickles[1]:
			gne = get_data_by_expression(vf_data,"^GNE")
			if ratio <= 0.91:
				lst = ['GNE_SNR_TKEO', 'GNE_NSR_TKEO']
				gne = add_variable_from_mean(gne, 'GNE_master', lst, True)
				gne.to_pickle("Pickles/Vocal/GNE_-91.pkl")
			else:
				gne.to_pickle("Pickles/Vocal/GNE_+91.pkl")
		else:
			if ratio <= 0.91:
				gne = pd.read_pickle("Pickles/Vocal/GNE_-91.pkl")
			else:
				gne = pd.read_pickle("Pickles/Vocal/GNE_+91.pkl")
		

		# VFER
		if pickles[2]:
			vfer = get_data_by_expression(vf_data,"^VFER")
			if ratio <= 0.98:
				vfer = delete_columns(vfer, ['VFER_entropy'])
				vfer.to_pickle("Pickles/Vocal/VFER_-98.pkl")
			else:
				vfer.to_pickle("Pickles/Vocal/VFER_+98.pkl")
		else:
			if ratio <= 0.98:
				vfer = pd.read_pickle("Pickles/Vocal/VFER_-98.pkl")
			else:
				vfer = pd.read_pickle("Pickles/Vocal/VFER_+98.pkl")

		"""

		if correlations:
			
			# GQ
			gq = get_data_by_expression(vf_data,"^GQ")
			group_correlation(gq)

			#GNE
			gne = get_data_by_expression(vf_data,"^GNE")
			lst = ['GNE_SNR_TKEO', 'GNE_NSR_TKEO']
			gne = add_variable_from_mean(gne, 'GNE_master', lst, False)
			group_correlation(gne)
			


	return vf_data

vf_data = vocal_fold(dic, 0.90, False, [1,0,0,1], True)


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
	"""
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
	"""
	return wf_data



def tqwt_features(dic):

	tqwt_data = get_data_from_dic(dic,"TQWT Features")

	"""
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

	"""
	return tqwt_data


#dic = general_dic(False)


#sum = 0

#bf_data = baseline_features(dic, 0.8, [0,0,0,0], [0,0,1,1], False)
#ip_data = intensity_parameters(dic, 0.90, [0,0,0], True)
#ff_data = formant_frequencies(dic, False)
#bp_data = bandwidth_parameters(dic, False)
#vf_data = vocal_fold(dic)
#mfcc_data = mfcc(dic)
#wf_data = wavelet_features(dic, False)
#tqwt_data = tqwt_features(dic)



#new_data = data[['id','gender']]
#new_data = pd.concat([new_data, bf_data, ip_data, ff_data, bp_data, vf_data, mfcc_data, wf_data, tqwt_data], axis=1, sort=False)


	



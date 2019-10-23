import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
import functions as func

register_matplotlib_converters()

data = pd.read_csv('../1_Dataset/pd_speech_features.csv', sep=',', decimal='.', skiprows=1)
print(data)

# --------------------------------------------------------
# 
#               B A S E L I N E   F E A T U R E S
# 
# --------------------------------------------------------

print('\n Baseline Features \n')
baseline = data.loc[:, 'PPE' : 'meanHarmToNoiseHarmonicity']
#Juntar class
baseline['Class'] = data['class']
print(baseline)

print('\n Filtered Baseline Features \n')
baseline_filt = baseline.drop(columns=['numPeriodsPulses', 'locPctJitter', 'locAbsJitter',
    'rapJitter', 'ppq5Jitter', 'locDbShimmer', 'apq3Shimmer', 'apq5Shimmer', 'apq11Shimmer', 
    'ddaShimmer'])
print(baseline_filt)

# Naive Bayes for Baseline Features non-filtered

y: np.ndarray = baseline.pop('Class').values
X: np.ndarray = baseline.values
labels: np.ndarray = pd.unique(y)

trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)

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
func.bar_chart(plt.gca(), xvalues, yvalues, 'Comparison of Naive Bayes Models Base-non-Filt', '', 'accuracy', percentage=True)


# Naive Bayes for Baseline Features filtered

y: np.ndarray = baseline_filt.pop('Class').values
X: np.ndarray = baseline_filt.values
labels: np.ndarray = pd.unique(y)

trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)

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
func.bar_chart(plt.gca(), xvalues, yvalues, 'Comparison of Naive Bayes Models Base-Filt', '', 'accuracy', percentage=True)

#plt.show()

# --------------------------------------------------------
# 
#                        M F C C
# 
# --------------------------------------------------------
#'''
print('\nMFCC \n')
mfcc = data.loc[:, 'mean_Log_energy' : 'std_12th_delta_delta']
mfcc['Class'] = data['class']
print(mfcc)

# The filtering data is based on the best attribute with the best correlation
# Mean has no correlation

print('\nMFCC with std deltas \n')
mfcc_std_delta = mfcc.drop(columns=["mean_Log_energy", "mean_MFCC_0th_coef", "mean_MFCC_1st_coef",
 "mean_MFCC_2nd_coef", "mean_MFCC_3rd_coef", "mean_MFCC_4th_coef", "mean_MFCC_5th_coef", 
 "mean_MFCC_6th_coef", "mean_MFCC_7th_coef", "mean_MFCC_8th_coef", "mean_MFCC_9th_coef", 
 "mean_MFCC_10th_coef", "mean_MFCC_11th_coef", "mean_MFCC_12th_coef", "mean_delta_log_energy",
 "mean_0th_delta", "mean_1st_delta", "mean_2nd_delta", "mean_3rd_delta", "mean_4th_delta",
 "mean_5th_delta", "mean_6th_delta", "mean_7th_delta", "mean_8th_delta", "mean_9th_delta",
 "mean_10th_delta", "mean_11th_delta", "mean_12th_delta", "mean_delta_delta_log_energy",
 "mean_delta_delta_0th", "mean_1st_delta_delta", "mean_2nd_delta_delta", "mean_3rd_delta_delta",
 "mean_4th_delta_delta", "mean_5th_delta_delta", "mean_6th_delta_delta", "mean_7th_delta_delta",
 "mean_8th_delta_delta", "mean_9th_delta_delta", "mean_10th_delta_delta", "mean_11th_delta_delta",
 "mean_12th_delta_delta", "std_Log_energy", "std_MFCC_0th_coef", "std_MFCC_1st_coef", 
 "std_MFCC_2nd_coef", "std_MFCC_3rd_coef", "std_MFCC_4th_coef", "std_MFCC_5th_coef", 
 "std_MFCC_6th_coef", "std_MFCC_7th_coef", "std_MFCC_8th_coef", "std_MFCC_9th_coef", 
 "std_MFCC_10th_coef", "std_MFCC_11th_coef", "std_MFCC_12th_coef", "std_delta_log_energy",
 "std_3rd_delta", "std_4th_delta", "std_5th_delta", "std_6th_delta", "std_7th_delta",
 "std_8th_delta", "std_10th_delta", "std_11th_delta", "std_12th_delta",
 "std_delta_delta_log_energy", "std_delta_delta_0th", "std_3rd_delta_delta",
 "std_4th_delta_delta", "std_5th_delta_delta",
 "std_6th_delta_delta", "std_7th_delta_delta", "std_8th_delta_delta", "std_9th_delta_delta",
 "std_10th_delta_delta", "std_11th_delta_delta", "std_12th_delta_delta"])
print(mfcc_std_delta)

print('\nMFCC with std deltas and coef \n')
mfcc_std_coef_deltas = mfcc.drop(columns=["mean_Log_energy", "mean_MFCC_0th_coef", "mean_MFCC_1st_coef",
 "mean_MFCC_2nd_coef", "mean_MFCC_3rd_coef", "mean_MFCC_4th_coef", "mean_MFCC_5th_coef", 
 "mean_MFCC_6th_coef", "mean_MFCC_7th_coef", "mean_MFCC_8th_coef", "mean_MFCC_9th_coef", 
 "mean_MFCC_10th_coef", "mean_MFCC_11th_coef", "mean_MFCC_12th_coef", "mean_delta_log_energy",
 "mean_0th_delta", "mean_1st_delta", "mean_2nd_delta", "mean_3rd_delta", "mean_4th_delta",
 "mean_5th_delta", "mean_6th_delta", "mean_7th_delta", "mean_8th_delta", "mean_9th_delta",
 "mean_10th_delta", "mean_11th_delta", "mean_12th_delta", "mean_delta_delta_log_energy",
 "mean_delta_delta_0th", "mean_1st_delta_delta", "mean_2nd_delta_delta", "mean_3rd_delta_delta",
 "mean_4th_delta_delta", "mean_5th_delta_delta", "mean_6th_delta_delta", "mean_7th_delta_delta",
 "mean_8th_delta_delta", "mean_9th_delta_delta", "mean_10th_delta_delta", "mean_11th_delta_delta",
 "mean_12th_delta_delta", 
 "std_MFCC_3rd_coef", "std_MFCC_4th_coef", 
 "std_MFCC_6th_coef", "std_MFCC_7th_coef", "std_MFCC_8th_coef", "std_MFCC_9th_coef", 
 "std_MFCC_10th_coef", "std_MFCC_11th_coef", "std_MFCC_12th_coef",
 "std_3rd_delta", "std_4th_delta", "std_5th_delta", "std_6th_delta", "std_7th_delta",
 "std_8th_delta", "std_10th_delta", "std_11th_delta", "std_12th_delta",
 "std_3rd_delta_delta",
 "std_4th_delta_delta", "std_5th_delta_delta", "std_6th_delta_delta",
 "std_7th_delta_delta", "std_8th_delta_delta", "std_9th_delta_delta",
 "std_10th_delta_delta", "std_11th_delta_delta", "std_12th_delta_delta"])
print(mfcc_std_coef_deltas)

print('\nMFCC filtered \n')
mfcc_filt = mfcc.drop(columns=[
 "std_MFCC_3rd_coef", "std_MFCC_4th_coef", 
 "std_MFCC_6th_coef", "std_MFCC_7th_coef", "std_MFCC_8th_coef", "std_MFCC_9th_coef", 
 "std_MFCC_10th_coef", "std_MFCC_11th_coef", "std_MFCC_12th_coef",
 "std_3rd_delta", "std_4th_delta", "std_5th_delta", "std_6th_delta", "std_7th_delta",
 "std_8th_delta", "std_10th_delta", "std_11th_delta", "std_12th_delta",
 "std_3rd_delta_delta",
 "std_4th_delta_delta", "std_5th_delta_delta", "std_6th_delta_delta",
 "std_7th_delta_delta", "std_8th_delta_delta", "std_9th_delta_delta",
 "std_10th_delta_delta", "std_11th_delta_delta", "std_12th_delta_delta"])
print(mfcc_filt)


# Naive Bayes for MFCC non-filtered

y: np.ndarray = mfcc.pop('Class').values
X: np.ndarray = np.abs(mfcc_filt.values) #GANDA Batota
labels: np.ndarray = pd.unique(y)

trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)

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
func.bar_chart(plt.gca(), xvalues, yvalues, 'Comparison of Naive Bayes Models MFCC-non-Filt', '', 'accuracy', percentage=True)


# Naive Bayes for MFCC filtered

y: np.ndarray = mfcc_filt.pop('Class').values
X: np.ndarray = np.abs(mfcc_filt.values) #GANDA Batota
labels: np.ndarray = pd.unique(y)

trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)

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
func.bar_chart(plt.gca(), xvalues, yvalues, 'Comparison of Naive Bayes Models MFCC-Filt', '', 'accuracy', percentage=True)


plt.show()


#'''
# --------------------------------------------------------
# 
#            I N T E N S I T Y   F E A T U R E S
# 
# --------------------------------------------------------
'''
print('\n Intensity Features \n')
Intensity = data.loc[:, 'minIntensity' : 'meanIntensity']
print(Intensity)
'''






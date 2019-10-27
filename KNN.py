import numpy as np

import pandas as pd
from pandas.plotting import register_matplotlib_converters

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier

import functions as func

register_matplotlib_converters()

data = pd.read_csv('../1_Dataset/pd_speech_features.csv', sep=',', decimal='.', skiprows=1)
print(data)

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


# KNN for Baseline Features - Non-Filtered

y: np.ndarray = baseline.pop('Class').values
X: np.ndarray = baseline.values
labels: np.ndarray = pd.unique(y)

trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)

nvalues = [1,3, 5, 7, 9, 11, 13, 15, 17, 19]
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
func.multiple_line_chart(plt.gca(), nvalues, values, 'KNN variants - Base-NonFilt', 'n', 'accuracy', percentage=True)

# KNN for Baseline Features - Filtered

#y: np.ndarray = baseline_filt.pop('Class').values
#X: np.ndarray = baseline_filt.values
#labels: np.ndarray = pd.unique(y)
#
#trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)
#
#nvalues = [1,3, 5, 7, 9, 11, 13, 15, 17, 19]
#dist = ['manhattan', 'euclidean', 'chebyshev']
#values = {}
#for d in dist:
#    yvalues = []
#    for n in nvalues:
#        knn = KNeighborsClassifier(n_neighbors=n, metric=d)
#        knn.fit(trnX, trnY)
#        prdY = knn.predict(tstX)
#        yvalues.append(metrics.accuracy_score(tstY, prdY))
#    values[d] = yvalues
#
#plt.figure()
#func.multiple_line_chart(plt.gca(), nvalues, values, 'KNN variants - Base-Filt', 'n', 'accuracy', percentage=True)


y: np.ndarray = baseline_filt.pop('Class').values
X: np.ndarray = baseline_filt.values
labels: np.ndarray = pd.unique(y)

trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)

nvalues = [100, 200, 300, 400]
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
func.multiple_line_chart(plt.gca(), nvalues, values, 'KNN variants - Base-Filt k*100', 'n', 'accuracy', percentage=True)

plt.show()
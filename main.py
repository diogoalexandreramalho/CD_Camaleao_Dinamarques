import pandas as pd
import numpy as np
from pandas.plotting import register_matplotlib_converters

import Balancing as smt
import Normalize as nor
import Decision_Tree as dt
#import Feature_Selection as filt

# Diferent strategys = ['minority', 'not majority', 'not minority', 'all' ]
strategy  = 'all'

data = pd.read_csv('pd_speech_features.csv', sep=',', decimal='.', skiprows=1)

norm_data = nor.run(data) #This is a copy
smote_norm_data = smt.run(norm_data, strategy, 42) #This is another copy
#smote_data = smt.run(data, 'minority', 42)

#filtered_data = filt.run(data)

print('Data:\n', data)
print('Norm Data:\n', norm_data)
print('Smote + Norm Data:\n', smote_norm_data)
#print('Smote Data:\n', smote_data)
print('Data:\n', data)

dt.decision_tree(smote_norm_data)




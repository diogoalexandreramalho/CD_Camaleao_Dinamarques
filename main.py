import pandas as pd
import numpy as np
from pandas.plotting import register_matplotlib_converters

import Balancing as smt
import Normalize as nor

data = pd.read_csv('pd_speech_features.csv', sep=',', decimal='.', skiprows=1)

norm_data = nor.run(data) #This is a copy
smote_norm_data = smt.run(norm_data, 'minority', 42) #This is another copy
smote_data = smt.run(data, 'minority', 42)

print('Data:\n', data)
print('Norm Data:\n', norm_data)
print('Smote + Norm Data:\n', smote_norm_data)
print('Smote Data:\n', smote_data)
print('Data:\n', data)




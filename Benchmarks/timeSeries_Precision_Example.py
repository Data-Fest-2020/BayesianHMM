# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 16:18:39 2020

@author: schep
"""

import numpy as np
import sklearn
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import math

# clean and transform data
data = pd.read_csv('C:/Users/schep/OneDrive/Desktop/DataFest/Results/y_electric_screwdriver_HMM_80001_smooth0.csv') 

data = pd.DataFrame(data)

data = data.dropna()

y_hat = data.loc[: , "y_hat"]
y_true = data.loc[: , "y_true"]

y_hat = np.array(y_hat)
y_hat = y_hat.astype('int')

y_true = np.array(y_true)
y_true = y_true.astype('int')


## range-based performance metrics
results = TSMetric(metric_option="classic", alpha_r=0.9, cardinality="reciprocal", bias_p="flat", bias_r="flat").score(y_true, y_hat)

# standard precision, recall
precRec = classification_report(y_true, y_hat, output_dict=True)
precRec = pd.DataFrame(precRec).to_latex()

# confusion-matrix
cm = confusion_matrix(y_hat, y_true)
cm = pd.DataFrame(cm).to_latex()

# plotAnomalies
plotAnomalies(y_true, y_hat)






















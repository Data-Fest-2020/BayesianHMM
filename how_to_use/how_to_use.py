"""
Created on Sat Aug 29 11:13:10 2020

@author: sds
"""

# how to use e.g. in spyder



# import modules
import numpy as np
import hiddensmmodel


# load data: nparray without timestamp
T = 10000
data = np.loadtxt('example-data.txt')[:T]

# train model
model = hiddensmmodel.Hsmm(data)

# estimated state sequenece to evaluate
model.states

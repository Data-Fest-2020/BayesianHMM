"""
author: stefan.depperschmidt@gmail.com

how_to_use_plainhmm.py
"""

# how to use e.g. in spyder

import numpy as np
import plainhmm

# load data: nparray without timestamp
T = 2500
train_data = np.loadtxt('example-data.txt')[:T]

# test data
test_data = np.loadtxt('example-data.txt')[:T]

# predefine number of states to infer and train
model = plainhmm.Hmm(nparray=train_data, number_of_states=10)

# get states from train_data
model.predict(train_data)

# get states from test_data
model.predict(test_data)
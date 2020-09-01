"""
Created on Sat Aug 29 11:13:10 2020

@author: sds
"""

# import modules
import hiddensmmodel
import load_data
from matplotlib import pyplot as plt

# load data: nparray without timestamp
mytool = "electric_screwdriver"
# mytool = "pneumatic_screwdriver"
# mytool = "pneumatic_rivet_gun"
data = load_data.load_data(mytool)

# train model
model = hiddensmmodel.Hsmm(data[13])

y_hat = model.states
y_hat = y_hat[0]
y_true = data[14]

plt.plot(y_true[10000:15000], 'bo', markersize=0.5)
plt.show()

plt.plot(y_hat[10000:15000], 'bo', markersize=0.5)
plt.show()

plt.plot(y_true[1:5000], 'bo', markersize=0.5)
plt.show()

plt.plot(y_hat[1:5000], 'bo', markersize=0.5)
plt.show()

plt.plot(y_true[15000:20000], 'bo', markersize=0.5)
plt.show()

plt.plot(y_hat[15000:20000], 'bo', markersize=0.5)
plt.show()

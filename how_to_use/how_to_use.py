"""
Created on Sat Aug 29 11:13:10 2020

@author: sds
"""

# import modules
import hiddensmmodel
import load_data
from matplotlib import pyplot as plt
import pandas as pd

# load data: nparray without timestamp
#mytool = "electric_screwdriver"
#mytool = "pneumatic_screwdriver"
mytool = "pneumatic_rivet_gun"

data = load_data.load_data(mytool)

# train model
model = hiddensmmodel.Hsmm(data[13], Nmax = 100)

y_hat = model.states
y_hat = y_hat[0]
y_hat = pd.DataFrame(y_hat)
y_true = pd.DataFrame(data[14])
y_true = y_true.rename(columns={0: 'y_true'})
y_hat = y_hat.rename(columns={0: 'y_hat'})
y_hat = y_hat + 1500

y = pd.DataFrame()
y = pd.concat([y_true, y_hat], axis=1)

y['count'] = 1

y_manC = y.groupby(['y_true','y_hat']).count()

y_manC = y_manC.reset_index()

states_hat = y_manC['y_hat'].unique()

for state_hat in states_hat:
    idx = y_manC['y_hat'][y_manC['y_hat']==state_hat]
    idx = idx.reset_index()
    help1 = y_manC.iloc[idx['index'],:]
    maxS = help1.iloc[help1['count'].argmax()]
    y_hat = y_hat.replace({maxS['y_hat']:maxS['y_true']})    




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

plt.plot(y_true[40000:45000], 'bo', markersize=0.5)
plt.show()

plt.plot(y_hat[40000:45000], 'bo', markersize=0.5)
plt.show()

plt.plot(y_true[65000:70000], 'bo', markersize=0.5)
plt.show()

plt.plot(y_hat[65000:70000], 'bo', markersize=0.5)
plt.show()

plt.plot(y_true[135000:140000], 'bo', markersize=0.5)
plt.show()

plt.plot(y_hat[135000:140000], 'bo', markersize=0.5)
plt.show()

plt.plot(y_true, 'bo', markersize=0.5)
plt.show()

plt.plot(y_hat, 'bo', markersize=0.5)
plt.show()


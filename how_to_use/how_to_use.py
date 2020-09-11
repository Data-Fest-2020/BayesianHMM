"""
@author: matthias.gruber@outlook.com

how_to_use.py
"""

# import modules
import hiddensmmodel
import data_manipulation
from matplotlib import pyplot as plt
import pandas as pd

mytool = "electric_screwdriver"
#mytool = "pneumatic_screwdriver"
#mytool = "pneumatic_rivet_gun"

# load data (and smooth it)
data = data_manipulation.load_data(mytool = mytool, smooth = False,
                                   windowSize = 40)

# train model
model = hiddensmmodel.Hsmm(data[13][0:70000], Nmax = 15, trunc = 200)
# get the true labels of the training set
y_true_train = data[14][0:70000]
# get the predicted and true labels of the test set
y_hat_test_un = model.predict(data[13][70001:])
y_true_test = data[14][70001:]
# match the predicted labels to the true ones and smooth outliers out
y_hat_test = data_manipulation.map_states(model, y_hat_test_un, y_true_train,
                                          smooth = True, windowSize = 19)

y_true_test = pd.DataFrame(y_true_test)
y_hat_test = pd.DataFrame(y_hat_test)
y_hat_test = y_hat_test.rename(columns={0: 'y_hat'})
y_true_test = y_true_test.rename(columns={0: 'y_true'})
y_pneumatic_screwdriver = pd.concat([y_true_test, y_hat_test], axis=1)
y_pneumatic_screwdriver.to_csv('y_pneumatic_screwdriver.csv')


plt.plot(y_true_test[1:5000], 'bo', markersize=0.5)
plt.show()

plt.plot(y_hat_test[1:5000], 'bo', markersize=0.5)
plt.show()

plt.plot(y_hat_test_un[1:5000], 'bo', markersize=0.5)
plt.show()

plt.plot(y_true_test[10000:15000], 'bo', markersize=0.5)
plt.show()

plt.plot(y_hat_test[10000:15000], 'bo', markersize=0.5)
plt.show()

plt.plot(y_true_test[15000:20000], 'bo', markersize=0.5)
plt.show()

plt.plot(y_hat_test[15000:20000], 'bo', markersize=0.5)
plt.show()

plt.plot(y_true_test[20000:25000], 'bo', markersize=0.5)
plt.show()

plt.plot(y_hat_test[20000:25000], 'bo', markersize=0.5)
plt.show()

plt.plot(y_true_test, 'bo', markersize=0.5)
plt.show()

plt.plot(y_hat_test, 'bo', markersize=0.5)
plt.show()




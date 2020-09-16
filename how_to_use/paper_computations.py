"""
@author: matthias.gruber@outlook.com

how_to_use.py
"""

# import modules
import hiddensmmodel
import data_manipulation
import pandas as pd
from matplotlib import pyplot as plt

mytool = "electric_screwdriver"
#mytool = "pneumatic_screwdriver"
#mytool = "pneumatic_rivet_gun"

#load data (and smooth it)
data = data_manipulation.load_data(mytool = mytool, smooth = True,
                                   windowSize = 40)

# train model
model = hiddensmmodel.Hsmm(data[13][39:70000], Nmax = 15, trunc = 120)
# get the true labels of the training set
y_true_train = data[14][39:70000]
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
y_pneumatic_screwdriver.to_csv('y_electric_screwdriver_HSMM_70001_smooth40.csv')



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



mytool = "electric_screwdriver"
#mytool = "pneumatic_screwdriver"
#mytool = "pneumatic_rivet_gun"

#load data (and smooth it)
data = data_manipulation.load_data(mytool = mytool, smooth = False,
                                   windowSize = 40)

# train model
model = hiddensmmodel.Hsmm(data[13][39:70000], Nmax = 15, trunc = 120)
# get the true labels of the training set
y_true_train = data[14][39:70000]
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
y_pneumatic_screwdriver.to_csv('y_electric_screwdriver_HSMM_70001_smooth0.csv')





















#mytool = "electric_screwdriver"
mytool = "pneumatic_screwdriver"
#mytool = "pneumatic_rivet_gun"

#load data (and smooth it)
data = data_manipulation.load_data(mytool = mytool, smooth = True,
                                   windowSize = 40)

# train model
model = hiddensmmodel.Hsmm(data[13][39:70000], Nmax = 15, trunc = 120)
# get the true labels of the training set
y_true_train = data[14][39:70000]
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
y_pneumatic_screwdriver.to_csv('y_pneumatic_screwdriver_HSMM_70001_smooth40.csv')







#mytool = "electric_screwdriver"
mytool = "pneumatic_screwdriver"
#mytool = "pneumatic_rivet_gun"

#load data (and smooth it)
data = data_manipulation.load_data(mytool = mytool, smooth = False,
                                   windowSize = 40)

# train model
model = hiddensmmodel.Hsmm(data[13][39:70000], Nmax = 15, trunc = 120)
# get the true labels of the training set
y_true_train = data[14][39:70000]
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
y_pneumatic_screwdriver.to_csv('y_pneumatic_screwdriver_HSMM_70001_smooth0.csv')


















#mytool = "electric_screwdriver"
#mytool = "pneumatic_screwdriver"
mytool = "pneumatic_rivet_gun"

#load data (and smooth it)
data = data_manipulation.load_data(mytool = mytool, smooth = True,
                                   windowSize = 40)

# train model
model = hiddensmmodel.Hsmm(data[13][39:110000], Nmax = 15, trunc = 120)
# get the true labels of the training set
y_true_train = data[14][39:110000]
# get the predicted and true labels of the test set
y_hat_test_un = model.predict(data[13][110001:])
y_true_test = data[14][110001:]
# match the predicted labels to the true ones and smooth outliers out
y_hat_test = data_manipulation.map_states(model, y_hat_test_un, y_true_train,
                                          smooth = True, windowSize = 19)

y_true_test = pd.DataFrame(y_true_test)
y_hat_test = pd.DataFrame(y_hat_test)
y_hat_test = y_hat_test.rename(columns={0: 'y_hat'})
y_true_test = y_true_test.rename(columns={0: 'y_true'})
y_pneumatic_screwdriver = pd.concat([y_true_test, y_hat_test], axis=1)
y_pneumatic_screwdriver.to_csv('y_pneumatic_rivet_gun_HSMM_110001_smooth40.csv')







#mytool = "electric_screwdriver"
#mytool = "pneumatic_screwdriver"
mytool = "pneumatic_rivet_gun"

#load data (and smooth it)
data = data_manipulation.load_data(mytool = mytool, smooth = False,
                                   windowSize = 40)

# train model
model = hiddensmmodel.Hsmm(data[13][39:110000], Nmax = 15, trunc = 120)
# get the true labels of the training set
y_true_train = data[14][39:110000]
# get the predicted and true labels of the test set
y_hat_test_un = model.predict(data[13][110001:])
y_true_test = data[14][110001:]
# match the predicted labels to the true ones and smooth outliers out
y_hat_test = data_manipulation.map_states(model, y_hat_test_un, y_true_train,
                                          smooth = True, windowSize = 19)

y_true_test = pd.DataFrame(y_true_test)
y_hat_test = pd.DataFrame(y_hat_test)
y_hat_test = y_hat_test.rename(columns={0: 'y_hat'})
y_true_test = y_true_test.rename(columns={0: 'y_true'})
y_pneumatic_screwdriver = pd.concat([y_true_test, y_hat_test], axis=1)
y_pneumatic_screwdriver.to_csv('y_pneumatic_rivet_gun_HSMM_110001_smooth0.csv')





































import plainhmm


mytool = "electric_screwdriver"
#mytool = "pneumatic_screwdriver"
#mytool = "pneumatic_rivet_gun"

#load data (and smooth it)
data = data_manipulation.load_data(mytool = mytool, smooth = True,
                                   windowSize = 40)



# predefine number of states to infer and train
model = plainhmm.Hmm(nparray=data[13][39:70000], number_of_states=10)

y_true_test = data[14][70001:]
# get states from train_data
train_data = data[13][39:70000]
y_true_train = data[14][39:70000]
# get states from test_data
y_hat_test_un = model.predict(data[13][70001:])
# match the predicted labels to the true ones and smooth outliers out
y_hat_test = data_manipulation.map_states(model, y_hat_test_un, y_true_train,
                                          smooth = True, windowSize = 19,
                                          HMM = True, train_data = train_data)


y_true_test = pd.DataFrame(y_true_test)
y_hat_test = pd.DataFrame(y_hat_test)
y_hat_test = y_hat_test.rename(columns={0: 'y_hat'})
y_true_test = y_true_test.rename(columns={0: 'y_true'})
y_pneumatic_screwdriver = pd.concat([y_true_test, y_hat_test], axis=1)
y_pneumatic_screwdriver.to_csv('y_electric_screwdriver_HMM_70001_smooth40.csv')








mytool = "electric_screwdriver"
#mytool = "pneumatic_screwdriver"
#mytool = "pneumatic_rivet_gun"

#load data (and smooth it)
data = data_manipulation.load_data(mytool = mytool, smooth = False,
                                   windowSize = 40)






# predefine number of states to infer and train
model = plainhmm.Hmm(nparray=data[13][39:70000], number_of_states=10)

y_true_test = data[14][70001:]
# get states from train_data
train_data = data[13][39:70000]
y_true_train = data[14][39:70000]
# get states from test_data
y_hat_test_un = model.predict(data[13][70001:])
# match the predicted labels to the true ones and smooth outliers out
y_hat_test = data_manipulation.map_states(model, y_hat_test_un, y_true_train,
                                          smooth = True, windowSize = 19,
                                          HMM = True, train_data = train_data)


y_true_test = pd.DataFrame(y_true_test)
y_hat_test = pd.DataFrame(y_hat_test)
y_hat_test = y_hat_test.rename(columns={0: 'y_hat'})
y_true_test = y_true_test.rename(columns={0: 'y_true'})
y_pneumatic_screwdriver = pd.concat([y_true_test, y_hat_test], axis=1)
y_pneumatic_screwdriver.to_csv('y_electric_screwdriver_HMM_70001_smooth0.csv')

























#mytool = "electric_screwdriver"
mytool = "pneumatic_screwdriver"
#mytool = "pneumatic_rivet_gun"

#load data (and smooth it)
data = data_manipulation.load_data(mytool = mytool, smooth = True,
                                   windowSize = 40)



# predefine number of states to infer and train
model = plainhmm.Hmm(nparray=data[13][39:70000], number_of_states=10)

y_true_test = data[14][70001:]
# get states from train_data
train_data = data[13][39:70000]
y_true_train = data[14][39:70000]
# get states from test_data
y_hat_test_un = model.predict(data[13][70001:])
# match the predicted labels to the true ones and smooth outliers out
y_hat_test = data_manipulation.map_states(model, y_hat_test_un, y_true_train,
                                          smooth = True, windowSize = 19,
                                          HMM = True, train_data = train_data)


y_true_test = pd.DataFrame(y_true_test)
y_hat_test = pd.DataFrame(y_hat_test)
y_hat_test = y_hat_test.rename(columns={0: 'y_hat'})
y_true_test = y_true_test.rename(columns={0: 'y_true'})
y_pneumatic_screwdriver = pd.concat([y_true_test, y_hat_test], axis=1)
y_pneumatic_screwdriver.to_csv('y_pneumatic_screwdriver_HMM_70001_smooth40.csv')








#mytool = "electric_screwdriver"
mytool = "pneumatic_screwdriver"
#mytool = "pneumatic_rivet_gun"

#load data (and smooth it)
data = data_manipulation.load_data(mytool = mytool, smooth = False,
                                   windowSize = 40)






# predefine number of states to infer and train
model = plainhmm.Hmm(nparray=data[13][39:70000], number_of_states=10)

y_true_test = data[14][70001:]
# get states from train_data
train_data = data[13][39:70000]
y_true_train = data[14][39:70000]
# get states from test_data
y_hat_test_un = model.predict(data[13][70001:])
# match the predicted labels to the true ones and smooth outliers out
y_hat_test = data_manipulation.map_states(model, y_hat_test_un, y_true_train,
                                          smooth = True, windowSize = 19,
                                          HMM = True, train_data = train_data)


y_true_test = pd.DataFrame(y_true_test)
y_hat_test = pd.DataFrame(y_hat_test)
y_hat_test = y_hat_test.rename(columns={0: 'y_hat'})
y_true_test = y_true_test.rename(columns={0: 'y_true'})
y_pneumatic_screwdriver = pd.concat([y_true_test, y_hat_test], axis=1)
y_pneumatic_screwdriver.to_csv('y_pneumatic_screwdriver_HMM_70001_smooth0.csv')

















#mytool = "electric_screwdriver"
#mytool = "pneumatic_screwdriver"
mytool = "pneumatic_rivet_gun"

#load data (and smooth it)
data = data_manipulation.load_data(mytool = mytool, smooth = True,
                                   windowSize = 40)



# predefine number of states to infer and train
model = plainhmm.Hmm(nparray=data[13][39:110000], number_of_states=10)

y_true_test = data[14][110001:]
# get states from train_data
train_data = data[13][39:110000]
y_true_train = data[14][39:110000]
# get states from test_data
y_hat_test_un = model.predict(data[13][110001:])
# match the predicted labels to the true ones and smooth outliers out
y_hat_test = data_manipulation.map_states(model, y_hat_test_un, y_true_train,
                                          smooth = True, windowSize = 19,
                                          HMM = True, train_data = train_data)


y_true_test = pd.DataFrame(y_true_test)
y_hat_test = pd.DataFrame(y_hat_test)
y_hat_test = y_hat_test.rename(columns={0: 'y_hat'})
y_true_test = y_true_test.rename(columns={0: 'y_true'})
y_pneumatic_screwdriver = pd.concat([y_true_test, y_hat_test], axis=1)
y_pneumatic_screwdriver.to_csv('y_pneumatic_rivet_gun_HMM_110001_smooth40.csv')








#mytool = "electric_screwdriver"
#mytool = "pneumatic_screwdriver"
mytool = "pneumatic_rivet_gun"

#load data (and smooth it)
data = data_manipulation.load_data(mytool = mytool, smooth = False,
                                   windowSize = 40)






# predefine number of states to infer and train
model = plainhmm.Hmm(nparray=data[13][39:110000], number_of_states=10)

y_true_test = data[14][110001:]
# get states from train_data
train_data = data[13][39:110000]
y_true_train = data[14][39:110000]
# get states from test_data
y_hat_test_un = model.predict(data[13][110001:])
# match the predicted labels to the true ones and smooth outliers out
y_hat_test = data_manipulation.map_states(model, y_hat_test_un, y_true_train,
                                          smooth = True, windowSize = 19,
                                          HMM = True, train_data = train_data)


y_true_test = pd.DataFrame(y_true_test)
y_hat_test = pd.DataFrame(y_hat_test)
y_hat_test = y_hat_test.rename(columns={0: 'y_hat'})
y_true_test = y_true_test.rename(columns={0: 'y_true'})
y_pneumatic_screwdriver = pd.concat([y_true_test, y_hat_test], axis=1)
y_pneumatic_screwdriver.to_csv('y_pneumatic_rivet_gun_HMM_110001_smooth0.csv')
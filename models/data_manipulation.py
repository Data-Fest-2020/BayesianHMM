"""
@author: matthias.gruber@outlook.com

data_manipulation.py
"""
# load extern moduls
from datatools import Tool, Measurement, DataTypes, MeasurementDataReader
from datatools import ACC, GYR, MAG, MIC
import pandas as pd
import numpy as np

def load_data(mytool, smooth = False, windowSize = 10):
    '''
    this instance is useful to load the data of the datatools package
    and manipulate it
    '''
    # load data of datatools package
    source = "./tool-tracking-data/"
    mdr = MeasurementDataReader(source=source)
    q = mdr.query(query_type=Measurement)
    data_dict = q.filter_by(Tool == mytool,
                            DataTypes == [ACC, GYR, MIC, MAG]).get()
    # res is a list which will be returned
    res = [mytool]
    # load all measurement_campaigns and downsample all of them to the mag
    for measurement_campaign in ["01","02","03","04"]:
        data = data_dict[measurement_campaign]
        acc = data.acc
        gyr = data.gyr
        mag = data.mag
        mic = data.mic
        data = pd.merge_asof(acc,gyr,
                             left_on="time [s]",
                             right_on="time [s]",
                             direction="nearest")

        data = pd.merge_asof(data,mic,
                             left_on="time [s]",
                             right_on="time [s]",
                             direction="nearest")

        data = pd.merge_asof(data,mag,
                             left_on="time [s]",
                             right_on="time [s]",
                             direction="nearest")
        # name and delete columns
        y_true = data["label_x"]
        time = data["time [s]"]
        y_true = y_true.iloc[:, 0]
        del data["label_x"]
        del data["label_y"]
        del data["time [s]"] # 102.292 timesteps are one second
        # append the data to our list
        res.append(data)
        res.append(y_true)
        res.append(time)        
    # the last list entry contains all concatenated measurement_campaigns 
    res.append(np.concatenate((res[1],res[4],res[7],res[10])))
    res.append(np.concatenate((res[2],res[5],res[8],res[11])))
    res.append(np.concatenate((res[3],res[6],res[9],res[12])))
    # smooth data by taking the mean of the windows and drop NANs
    if smooth == True:
        for measurement_campaign in [1,4,7,10,13]:
            res[measurement_campaign] = pd.DataFrame(res[measurement_campaign])
            res[measurement_campaign] = res[measurement_campaign].rolling(window=windowSize).mean()#.dropna(axis = 0)
    # return the list with all 4 measurement_campaigns, true labels, time stemps
    # and as last entry concatenate all 4 measurement_campaigns
    return res

def map_states(model, y_hat_test, y_true_train, smooth = True, windowSize = 199, HMM = False, train_data = False):
    '''
    this instance is useful to map the predicted labels to the true ones
    as the hsmm model is discovering the number of the labels by itself,
    we have to search for the most matches and rename the labels to the true
    ones according to the train set
    '''
    if HMM == False:
    # get predictet and true labels of the train set
        y_hat_train = model.states
        y_hat_train = y_hat_train[0]
    if HMM == True:
    # get predictet and true labels of the train set
        y_hat_train = model.predict(train_data)
    y_hat_train = pd.DataFrame(y_hat_train)
    y_true_train = pd.DataFrame(y_true_train)
    y_hat_test = pd.DataFrame(y_hat_test)
    y_true_train = y_true_train.rename(columns={0: 'y_true'})
    y_hat_train = y_hat_train.rename(columns={0: 'y_hat'})
    # add huge number to the predicted labels of the train set as the function fails
    # if the predicted ones matches with the true labels
    y_hat_train = y_hat_train + 1500
    y_hat_test = y_hat_test + 1500
    y_train = pd.DataFrame()
    y_train = pd.concat([y_true_train, y_hat_train], axis=1)
    # add auxiliary variable the the data frame in order to count the matches
    y_train['count'] = 1
    # search for the matches
    y_manC = y_train.groupby(['y_true','y_hat']).count()
    y_manC = y_manC.reset_index()
    states_hat = y_manC['y_hat'].unique()
    # rename the predicted labels of the train set to the true ones with the most matches
    for state_hat in states_hat:
        idx = y_manC['y_hat'][y_manC['y_hat']==state_hat]
        idx = idx.reset_index()
        help1 = y_manC.iloc[idx['index'],:]
        maxS = help1.iloc[help1['count'].argmax()]
        y_hat_test = y_hat_test.replace({maxS['y_hat']:maxS['y_true']})   
    # smooth data by taking the median of the windows and drop NANs
    if smooth == True:   
        y_hat_test = y_hat_test.rolling(window=windowSize).quantile(quantile=0.65, interpolation = "nearest")#.dropna(axis = 0)
    # return the manipulated predicted labels
    return y_hat_test

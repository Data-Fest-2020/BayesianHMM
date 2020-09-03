#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 06:00:06 2020

@author: matthias
"""

from datatools import Tool, Measurement, DataTypes, MeasurementDataReader
from datatools import ACC, GYR, MAG, MIC
import pandas as pd
import numpy as np

def load_data(mytool):
    source = "./tool-tracking-data/"
    mdr = MeasurementDataReader(source=source)
    q = mdr.query(query_type=Measurement)
    data_dict = q.filter_by(Tool == mytool,
                            DataTypes == [ACC, GYR, MIC, MAG]).get()
    res = [mytool]
    for measurement_campaign in ["01","02","03","04"]:
        data = data_dict[measurement_campaign]
        acc = data.acc
        gyr = data.gyr
        mag = data.mag
        mic = data.mic
        data = pd.merge_asof(mag,gyr,
                             left_on="time [s]",
                             right_on="time [s]",
                             direction="nearest")

        data = pd.merge_asof(data,mic,
                             left_on="time [s]",
                             right_on="time [s]",
                             direction="nearest")

        data = pd.merge_asof(data,acc,
                             left_on="time [s]",
                             right_on="time [s]",
                             direction="nearest")
        y_true = data["label_x"]
        time = data["time [s]"].to_numpy()
        y_true = y_true.iloc[:, 0].to_numpy()
        del data["label_x"]
        del data["label_y"]
        del data["time [s]"] # 102.292 timesteps are one second
        res.append(data.to_numpy())
        res.append(y_true)
        res.append(time)
        
    res.append(np.concatenate((res[1],res[4],res[7],res[10])))
    res.append(np.concatenate((res[2],res[5],res[8],res[11])))
    res.append(np.concatenate((res[3],res[6],res[9],res[12])))
    return res
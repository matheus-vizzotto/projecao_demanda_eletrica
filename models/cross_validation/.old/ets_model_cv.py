# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 22:42:44 2022

@author: Matheus
"""

import pandas as pd
import numpy as np
from load import load_data
from load import train_test_split
from load import get_measures
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from sktime.forecasting.ets import AutoETS
import warnings 
# configs
warnings.filterwarnings('ignore') # remove warnings
plt.style.use('fivethirtyeight')
rcParams['figure.figsize'] = 15, 5


df = load_data()
df["load_mwmed"].interpolate(method = "linear", inplace = True) 

# split train - test
n_test = 15
train, test = train_test_split(df, n_test)

predictions = list()
train, test = train_test_split(df, n_test)
history = [x for x in train.load_mwmed]
for i in range(len(test)):
    forecaster = AutoETS(auto=True, n_jobs=-1, seasonal = "add", sp=7)
    forecaster.fit(pd.Series(history))
    load_fc = forecaster.predict(fh=[1]).iloc[0]
    predictions.append(load_fc)
    history.append(test.load_mwmed[i])
    print(f'>expected = {test.load_mwmed[i]}, predicted = {load_fc}')
    
    
plt.figure()
plt.plot(test.load_mwmed.reset_index(drop=True))
plt.plot(predictions)
plt.show()

measures = get_measures(pd.Series(predictions), test.load_mwmed)
df_measures = pd.DataFrame([measures])
print(df_measures)

y_hat = pd.Series(predictions)
y_hat.index.names = ["date"]
y_hat.index = test.index 
y_hat.columns = ["forecast"]
y_hat.to_csv("validation/ets_fc.csv")
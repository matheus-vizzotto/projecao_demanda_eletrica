# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 21:43:50 2022

@author: Matheus
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
from load import load_data
from load import series_to_supervised
from load import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from load import get_measures
from sklearn.tree import DecisionTreeRegressor
import warnings
warnings.filterwarnings("ignore")
rcParams['figure.figsize'] = 15, 5 # tamanho das figuras

def multi_step_forecast(data, lag, n):
    n_test = outs = n
    data = series_to_supervised(values, n_in = lag, n_out = outs, dropnan=False)
    train, test = train_test_split(data, n_test)
    train.dropna(inplace = True)
    response_vars = data.columns[-(outs):]
    predictions = list()
    for h, response in enumerate(response_vars):
        cols = [x for x in data.columns[:lag]]
        cols.append(response)
        data_ = train[cols]
        nrows = data_.shape[0]
        data_ = data_.iloc[:nrows-h, :] 
        data_X, data_y = data_.iloc[:, :-1], data_.iloc[:, -1]
        model = DecisionTreeRegressor(random_state = 0)
        model.fit(data_X, data_y)
        testX, testy = test.reset_index(drop=True).loc[0, :"var1(t-1)"], test.reset_index(drop=True).loc[0, response]
        pred = model.predict([testX])[0]
        print(f"Predicting {response}\n    > expected: {testy}, predicted: {pred}")
        predictions.append(pred)
    measures = get_measures(pd.Series(predictions), test["var1(t)"])
    df_measures = pd.DataFrame([measures])
    return predictions, df_measures, test

df = load_data()
df.interpolate(method = "linear", inplace = True)
values = df.values.tolist()

lags = 60
h = 15
pred, measures, test = multi_step_forecast(values, lags, h)
print(measures)


plt.figure()
plt.plot(test["var1(t)"].reset_index(drop = True), label = "test")
plt.plot(pred, label = "forecast")
plt.legend()
plt.show()

pred = pd.DataFrame(pred, columns = ["forecast"], index = df.iloc[-h:].index)
pred.to_csv("validation/decisiontrees_fc.csv")
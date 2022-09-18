# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 17:09:31 2022

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
import lightgbm as lgb

plt.style.use('fivethirtyeight') # estilo dos gráficos
rcParams['figure.figsize'] = 15, 5 # tamanho das figuras

def lightgbm_forecast(train, testX):
	# transform list into array
	train = np.asarray(train)
	# split into input and output columns
	trainX, trainy = train[:, :-1], train[:, -1]
	# fit model
	model = lgb.LGBMRegressor(objective='regression', n_estimators=1000)
	model.fit(trainX, trainy)
	# make a one-step prediction
	yhat = model.predict([testX])
	return yhat[0]

# walk-forward validation for univariate data
def walk_forward_validation(data, n_test):
    predictions = list()
    # split dataset
    train, test = train_test_split(data, n_test)
    # seed history with training dataset
    history = [x for x in train]
    # step over each time-step in the test set
    for i in range(len(test)):
        # split test row into input and output columns
        testX, testy = test[i, :-1], test[i, -1]
        # fit model on history and make a prediction
        yhat = lightgbm_forecast(history, testX)
        # store forecast in list of predictions
        predictions.append(yhat)
        # add actual observation to history for the next loop
        history.append(test[i])
        # summarize progress
        print('>expected = %.1f, predicted = %.1f' % (testy, yhat))
    # estimate prediction error
    mae = mean_absolute_error(test[:, -1], predictions)
    mape = mean_absolute_percentage_error(test[:, -1], predictions)
    rmse = np.sqrt(mean_squared_error(test[:, -1], predictions))    
    return mae, mape, rmse, test[:, -1], predictions


df_load = load_data()
df_weather = pd.read_csv("../../data/weather_daily_data.csv", parse_dates=["DATA"])

# gets the same period for both dataframes
df_weather = df_weather[df_weather.DATA.isin(df_load.index)]
df_load = df_load[df_weather.DATA.min():df_weather.DATA.max()] 


df_load_2 = df_load.reset_index()
df_merged = pd.merge(df_weather, df_load_2, left_on = "DATA", right_on = "date", how = "outer")
df_merged.drop("date", axis = 1, inplace = True)

df_merged.dropna(how = "all", inplace = True)
df_merged.sort_values(by = "DATA", inplace = True)
df_merged.load_mwmed = df_merged.load_mwmed.interpolate(method="linear")

df_load_3 = df_merged.load_mwmed
values = df_load_3.values.tolist()
# Define the number of lag observations as input (X)
lag = 15 #ou 60
data1 = series_to_supervised(values, n_in = lag)
data2 = pd.DataFrame()
for col in df_weather.columns:
    if col == "DATA":
        continue
    else:
        values = df_weather[col].values.tolist()
        df_ = series_to_supervised(values, n_in = lag)
        df_.drop("var1(t)", axis = 1, inplace = True) # the response variable is the load dataframe
        df_.columns = [f"{x}_{col}" for x in df_.columns]
        data2 = pd.concat([data2, df_], axis = 1)
        
df_weather_load = pd.concat([data2, data1], axis = 1)

n_test = 15
mae, mape, rmse, y, yhat = walk_forward_validation(df_weather_load.values, n_test)

# plot expected vs predicted
plt.plot(y, label = 'Expected')
plt.plot(yhat, label = 'Predicted', color = 'orange')
plt.legend()
plt.show()

measures = get_measures(pd.Series(yhat), pd.Series(y))
df_measures = pd.DataFrame([measures])
print(df_measures)

fc = pd.DataFrame(list(zip(df_load.index[-n_test:], yhat)), columns = ["date", "forecast"])
fc.set_index("date", inplace = True)
fc.to_csv("validation/lightgbm_multivariate_fc.csv")
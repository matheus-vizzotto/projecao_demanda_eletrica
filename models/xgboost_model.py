# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 22:23:09 2022

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
from xgboost import XGBRegressor

####################### configurações de ambiente
import warnings 
warnings.filterwarnings('ignore') # retirar avisos
plt.style.use('fivethirtyeight')  # estilo do gráfico
#######################


def xgboost_forecast(train, testX):
	# transform list into array
	train = np.asarray(train)
	# split into input and output columns
	trainX, trainy = train[:, :-1], train[:, -1]
	# fit model
	model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
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
        yhat = xgboost_forecast(history, testX)
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


df = load_data()
values = df.values.tolist()
# Define the number of lag observations as input (X)
lag = 60 
data = series_to_supervised(values, n_in = lag)
data.head(5)

mae, mape, rmse, y, yhat = walk_forward_validation(data.values, 31)

# plot expected vs predicted
plt.plot(y, label = 'Expected')
plt.plot(yhat, label = 'Predicted', color = 'orange')
plt.legend()
plt.show()

measures = get_measures(pd.Series(yhat), pd.Series(y))
df_measures = pd.DataFrame([measures])
print(df_measures)
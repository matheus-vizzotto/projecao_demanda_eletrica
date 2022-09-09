# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 22:22:52 2022

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
import xgboost as xgb
from sktime.forecasting.compose import make_reduction


import warnings
warnings.filterwarnings("ignore")
rcParams['figure.figsize'] = 15, 5 # tamanho das figuras


df = load_data()
df.interpolate(method = "linear", inplace = True)
df_sk = df.asfreq('D')
n_test = 15
X, y = train_test_split(df_sk, n_test)

regressor = lgb.LGBMRegressor(objective='regression', n_estimators=1000)
#objective loss functions: "reg:squarederror",“reg:squaredlogerror“, “reg:logistic“, “reg:pseudohubererror“, “reg:gamma“, and “reg:tweedie“
forecaster = make_reduction(regressor, window_length=60, strategy="recursive")
forecaster.fit(X)   # fit 
predictions = forecaster.predict(fh = [x for x in range(1,n_test + 1)]) 

measures = get_measures(predictions.load_mwmed, y.load_mwmed)
df_measures = pd.DataFrame([measures])
print(df_measures)

plt.figure()
plt.plot(y, label = "test")
plt.plot(predictions, label = "forecast")
plt.legend()
plt.show()

predictions.index.names = ["date"]
predictions.to_csv("validation/univariate_lightgbm_seq_fc.csv")



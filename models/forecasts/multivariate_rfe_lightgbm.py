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
from sklearn.feature_selection import RFE
from load import get_measures
import lightgbm as lgb

import warnings
warnings.filterwarnings("ignore")
rcParams['figure.figsize'] = 15, 5 # tamanho das figuras



def multi_step_forecast(data, n):
    n_test = outs = n
    train, test = train_test_split(data, n_test)
    train.dropna(inplace = True)
    response_vars = data.columns[-(outs):]
    predictions = list()
    h_var = dict()
    for h, response in enumerate(response_vars):
        cols = [x for x in data.columns[:data.shape[1] - outs]]
        cols.append(response)
        data_ = train[cols]
        nrows = data_.shape[0]
        data_ = data_.iloc[:nrows-h, :] 
        data_X, data_y = data_.iloc[:, :-1], data_.iloc[:, -1]
        estimator = lgb.LGBMRegressor(objective='regression', n_estimators=1000)
        selector = RFE(estimator, n_features_to_select=20, step=1)
        print(f"selecting features...")
        selector = selector.fit(data_X, data_y)
        selected_vars = [x for x in data_X.columns[selector.support_]] 
        h_var[f"h_{h}"] = selected_vars
        testX, testy = test.reset_index(drop=True).loc[0, :"var1(t-1)"], test.reset_index(drop=True).loc[0, response]
        pred = selector.predict([testX])[0]
        print(f"Predicting {response}\n  > expected: {testy}, predicted: {pred}")
        predictions.append(pred)
    measures = get_measures(pd.Series(predictions), test["var1(t)"])
    df_measures = pd.DataFrame([measures])
    return predictions, df_measures, test, h_var

df_load = load_data()
df_weather = pd.read_csv("../../data/weather_daily_data.csv", parse_dates=["DATA"])
df_weather = df_weather[df_weather.DATA.isin(df_load.index)]
df_load = df_load[df_weather.DATA.min():df_weather.DATA.max()] 



df_load_2 = df_load.reset_index()
df_merged = pd.merge(df_weather, df_load_2, left_on = "DATA", right_on = "date", how = "outer")
df_merged.drop("date", axis = 1, inplace = True)

df_merged.dropna(how = "all", inplace = True)
df_merged.sort_values(by = "DATA", inplace = True)
df_merged.load_mwmed = df_merged.load_mwmed.interpolate(method="linear")


lag = 15    # lags das variáveis climáticas e de carga que serão utilizadas
n_test = outs = 15 # horizonte de previsão e teste (= número de modelos)

df_load_3 = df_merged.load_mwmed
df_load_3.index = df_merged.DATA
df_load_3 = df_load_3["2008-01-01":] # TESTE: MAPE PASSOU DE 3,2 PARA 3,0
values = df_load_3.values.tolist()
data1 = series_to_supervised(values, n_in = lag, n_out=outs, dropnan=False)

data2 = pd.DataFrame()
df_weather.set_index("DATA", inplace=True) # TESTE: MAPE PASSOU DE 3,2 PARA 3,0
df_weather = df_weather["2008-01-01":] # TESTE: MAPE PASSOU DE 3,2 PARA 3,0
for col in df_weather.columns:
    if col == "DATA":
        continue
    else:
        values = df_weather[col].values.tolist()
        df_ = series_to_supervised(values, n_in = lag, dropnan=False)
        df_.drop("var1(t)", axis = 1, inplace = True) # the response variable is the load dataframe
        df_.columns = [f"{x}_{col}" for x in df_.columns]
        data2 = pd.concat([data2, df_], axis = 1)
        
df_weather_load = pd.concat([data2, data1], axis = 1)


pred, measures, test, h_var = multi_step_forecast(df_weather_load, outs)
print(measures)

pred = pd.DataFrame(pred, columns = ["forecast"], index = df_load.iloc[-n_test:].index)
pred.to_csv("validation/multivariate_rfe_lightgbm_fc.csv")
with open('validation/selected_vars.json', 'w') as f:
    json.dump(h_var, f)



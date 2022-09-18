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
from collections import defaultdict
import json
import warnings
warnings.filterwarnings("ignore")
rcParams['figure.figsize'] = 15, 5

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

lag = 15
outs = n_test = 15
df_load_3 = df_merged.load_mwmed
df_load_3.index = df_merged.DATA
df_load_3 = df_load_3["2008-01-01":]
values = df_load_3.values.tolist()
data1 = series_to_supervised(values, n_in = lag, n_out=outs, dropnan=False)
data2 = pd.DataFrame()
df_weather.set_index("DATA", inplace=True) # TESTE: MAPE PASSOU DE 3,2 PARA 3,0
df_weather = df_weather["2008-01-01":] 
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

lag = 15
outs = n_test = horz = 15
df_load_3 = df_merged.load_mwmed
df_load_3.index = df_merged.DATA
df_load_3 = df_load_3["2008-01-01":]
values = df_load_3.values.tolist()
data1 = series_to_supervised(values, n_in = lag, n_out=outs, dropnan=False)
data2 = pd.DataFrame()
df_weather = df_weather["2008-01-01":] 
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


folds = 15 #partições
rows = df_weather_load.shape[0]
out = defaultdict(dict)
for fold in range(folds,0,-1):
    slide = rows-((fold-1)*horz)
    df_cv = df_weather_load.iloc[:slide,:]
    #print(df_cv.tail())
    
    train, test = train_test_split(df_cv, n_test)
    train.dropna(inplace = True)
    response_vars = df_weather_load.columns[-(outs):]
    print(f"predicting for cv {fold}...")
    predictions = list()
    for h, response in enumerate(response_vars):
        cols = [x for x in df_cv.columns[:df_cv.shape[1] - outs]]
        cols.append(response)
        data_ = train[cols]
        nrows = data_.shape[0]
        data_ = data_.iloc[:nrows-h, :] 
        data_X, data_y = data_.iloc[:, :-1], data_.iloc[:, -1]
        model = lgb.LGBMRegressor(objective='regression', n_estimators=1000)
        model.fit(data_X, data_y)
        #print(data_X)
        testX, testy = test.reset_index(drop=True).loc[0, :"var1(t-1)"], test.reset_index(drop=True).loc[0, response]
        pred = model.predict([testX])[0]
        print(f"\tPredicting {response}\n\t\t> expected: {testy}, predicted: {pred}")
        predictions.append(pred)
    out[f"cv_{fold}"]["pred"] = predictions
    out[f"cv_{fold}"]["test"] = test["var1(t)"].to_list()
d = dict(out)


mapes = []
for x in d:
    meas = get_measures(pd.Series(d[x]["pred"]),pd.Series(d[x]["test"]))
    print(meas)
    mapes.append(meas["mape"])

with open('validation/multivariate_lightgbm_cv.json', 'w') as f:
    json.dump(d, f)
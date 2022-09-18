import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from load import load_data
from load import train_test_split
from load import get_measures
import fbprophet 
from collections import defaultdict
import json
import warnings # retirar avisos
warnings.filterwarnings('ignore')

df = load_data().reset_index()
df.load_mwmed = df.load_mwmed.interpolate(method="linear")
df.columns = ['ds','y']

folds = 15 #partições
horz = n_test = 15 #horizonte de predição
rows = df.shape[0]
out = defaultdict(dict)
df_base = pd.DataFrame()
for fold in range(folds,0,-1):
    print(f"forecasting fold {fold}...")
    #slide = rows-(fold*horz)#-1
    slide = rows-((fold-1)*horz)
    df_cv = df.iloc[:slide]
    train, test = train_test_split(df_cv, n_test)
    train.columns = ['ds', 'y']
    m = fbprophet.Prophet(daily_seasonality=True)
    model = m.fit(train)
    future = m.make_future_dataframe(periods = 15, freq = 'D')
    prediction = m.predict(future)
    pred_ = prediction[['ds','yhat']].iloc[-15:]
    pred_["ult_dt_train"] = df_cv.ds.max()
    df_base = pd.concat([df_base, pred_], axis = 0)
    out[f"cv_{fold}"]["pred"] = pred_["yhat"].to_list()
    out[f"cv_{fold}"]["test"] = test["y"].to_list()
d = dict(out)

mapes = []
for x in d:
    meas = get_measures(pd.Series(d[x]["pred"]),pd.Series(d[x]["test"]))
    print(meas)
    mapes.append(meas["mape"])

print(np.mean(mapes))

with open('validation/prophet_cv.json', 'w') as f:
    json.dump(d, f)
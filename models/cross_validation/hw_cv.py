import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing, Holt
from load import train_test_split
from load import load_data
from load import get_measures
from collections import defaultdict
import json
import warnings # retirar avisos
warnings.filterwarnings('ignore')

df = load_data()
df.load_mwmed = df.load_mwmed.interpolate(method="linear")

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
    n_test = 15
    train, test = train_test_split(df_cv, n_test)
    fit1 = ExponentialSmoothing(train ,seasonal_periods=7,trend='add', seasonal='mul').fit() # seasonal='mul' é melhor
    y_hat = fit1.forecast(n_test)
    out[f"cv_{fold}"]["pred"] = y_hat.to_list()
    out[f"cv_{fold}"]["test"] = test["load_mwmed"].to_list()
d = dict(out)

mapes = []
for x in d:
    meas = get_measures(pd.Series(d[x]["pred"]),pd.Series(d[x]["test"]))
    print(meas)
    mapes.append(meas["mape"])

print(np.mean(mapes))

with open('validation/hw_cv.json', 'w') as f:
    json.dump(d, f)
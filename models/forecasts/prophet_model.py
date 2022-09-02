# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 22:52:05 2022

@author: Matheus
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from load import get_measures
from load import load_data
from load import train_test_split
import fbprophet

####################### configurações de ambiente
import warnings 
warnings.filterwarnings('ignore') # retirar avisos
plt.style.use('fivethirtyeight')  # estilo do gráfico
#######################

def create_future(start, t, cal_vars = False):
    """ Função para criar DataFrame de datas (dias) seguintes a T, assim como as variáveis de calendário se cal_vars = True.
       start: T + 1
       t: períodos à frente """
    dates = pd.date_range(start, freq = 'd', periods = t)
    df = pd.DataFrame(dates, columns = ['t'])
    if cal_vars == True:
        df = create_features(df, 't')
    elif cal_vars == False:
        pass
    return df

df = load_data()
n_test = 10
train, test = train_test_split(df, n_test)
train.reset_index(inplace = True)
train.columns = ['ds', 'y']  # IMPORTANTE: RENOMEAR COLUNA DE DATA E DE OBSERVAÇÕES PARA O PROPHET

model = fbprophet.Prophet(daily_seasonality=True)
model.fit(train)

future = create_future(test.index[0], len(test))    # cria dataframe de datas futuras
future.columns = ['ds']

forecast = model.predict(future)

plt.figure(figsize=(20,5))
plt.plot(forecast.ds, forecast.yhat, label = "forecast")    # forecast.ds = data no eixo x; forecast.yhat = forecast no eixo y
plt.plot(test.load_mwmed, label = "observed")   # test já tem data como índice, então não precisa especificar o eixo x
plt.legend()
plt.title(f"Prophet forecast (h={n_test})")
plt.show()

measures = get_measures(forecast.yhat, test)
df_measures = pd.DataFrame([measures])
print(df_measures)

fc = pd.DataFrame(list(zip(forecast.ds, forecast.yhat)), columns = ["date", "forecast"])
fc.set_index("date", inplace = True)
fc.to_csv("validation/prophet_fc.csv")
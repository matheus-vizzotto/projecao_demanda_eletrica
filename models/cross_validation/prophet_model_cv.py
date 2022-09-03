# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 22:58:11 2022

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
    return df

df = load_data()

n_test = 15

train, test = train_test_split(df, n_test)
train.reset_index(inplace = True)
train.columns = ['ds', 'y']  # IMPORTANTE: RENOMEAR COLUNA DE DATA E DE OBSERVAÇÕES PARA O PROPHET

predictions = list()
history = train.copy(deep=True)
for i in range(len(test)):
    model = fbprophet.Prophet() # daily_seasonality=True
    model.fit(history)
    future = create_future(test.index[i], 1)    # cria dataframe de datas futuras
    future.columns = ['ds']
    forecast = model.predict(future)
    load_fc = forecast.yhat[0]
    predictions.append(load_fc)
    new_line = forecast[["ds", "yhat"]]
    history = pd.concat([history, new_line], axis = 0)
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
y_hat.to_csv("validation/prophet_fc.csv")
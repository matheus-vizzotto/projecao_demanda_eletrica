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
    if cal_vars == True:
        df = create_features(df, 't')
    elif cal_vars == False:
        pass
    return df

df = load_data()

n_test = 15

train, test = train_test_split(df, n_test)
train.reset_index(inplace = True)
train.columns = ['ds', 'y']  # IMPORTANTE: RENOMEAR COLUNA DE DATA E DE OBSERVAÇÕES PARA O PROPHET

predictions = list()
history = [x for x in train.load_mwmed]
for i in range(len(test)):
    model = fbprophet.Prophet(daily_seasonality=True)
    model.fit(history)
    load_fc = SARIMA_model.forecast(1)[0]
    predictions.append(load_fc)
    history.append(test.load_mwmed[i])
    print(f'>expected = {test.load_mwmed[i]}, predicted = {load_fc}')
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 00:20:41 2022

@author: Matheus
"""

import pandas as pd
#import numpy as np
from load import load_data
from load import train_test_split
#from scipy.special import boxcox, inv_boxcox
from scipy.stats import boxcox
from scipy.special import inv_boxcox
import pmdarima as pm
from load import get_measures
import matplotlib.pyplot as plt
import warnings 

####################### configurações de ambiente
warnings.filterwarnings('ignore') # retirar avisos
plt.style.use('fivethirtyeight')
#######################

# carrega dados
df = load_data()
df["load_mwmed"].interpolate(method = "linear", inplace = True)  # preenche valores vazios
#df = boxcox(df, 2.5)
bc = boxcox(df)
df = bc[0]
lambda_ = bc[1]

# split treino-teste
n_test = 31
train, test = train_test_split(df, n_test)


# auto-arima 
# (Best model boxcox:  ARIMA(4,1,2)(1,0,1)[7] intercept)
SARIMA_model = pm.auto_arima(train, 
                            start_p=1, 
                            start_q=1,
                            test='kpss', 
                            max_p=4, max_q=5, # maximum p and q
                            max_P=1, max_Q=1,
                            #d=None,
                            max_d = 2,
                            #D=None# let model determine 'd'
                            max_D = 1,
                            seasonal=True, 
                            m=7, # frequency of series (if m==1, seasonal is set to FALSE automatically)
                            trace=True, #logs 
                            error_action='warn', #shows errors ('ignore' silences these)
                            suppress_warnings=True,
                            stepwise=True,
                            information_criterion='bic')


# parâmetros
print(SARIMA_model.summary())

# diagnósticos dos resíduos
SARIMA_model.plot_diagnostics(figsize=(15,5))
plt.subplots_adjust(hspace= 0.7)
plt.show()

# forecast
fc = pd.Series(SARIMA_model.predict(n_periods=n_test)) # transforma forecast em Series
fc.index = test.index # deixa o forecast e o teste com os mesmo índices para plotar

# medidas de acurácia
#fc = inv_boxcox(fc, 2.5) # volta a escala para o original
fc = inv_boxcox(fc, lambda_)
test = inv_boxcox(test, 2.5)
medidas_fc = get_measures(fc, test) 
df_medidas_fc = pd.DataFrame([medidas_fc])
print(df_medidas_fc)

# write forecst csv
#fc.to_csv("validation/auto_arima_fc.csv")

# visualização do forecast
plt.figure(figsize = (15, 5))
plt.plot(fc, c = "red", label = "forecast")
plt.plot(test, c = "blue", label = "actual")
plt.legend()
plt.show()



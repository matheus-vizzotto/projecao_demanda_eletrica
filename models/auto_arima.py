# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 00:20:41 2022

@author: Matheus
"""

import pandas as pd
import numpy as np
from load import load_data
from load import train_test_split
#from scipy.special import boxcox, inv_boxcox
#from scipy.stats import boxcox
#from scipy.special import inv_boxcox
import pmdarima as pm
from load import get_measures
import matplotlib.pyplot as plt
import warnings 
#import statsmodels.api as sm

####################### configurações de ambiente
warnings.filterwarnings('ignore') # retirar avisos
plt.style.use('fivethirtyeight')
#######################

# carrega dados
df = load_data()
df["load_mwmed"].interpolate(method = "linear", inplace = True)  # preenche valores vazios
#bc = boxcox(df.load_mwmed)
#df.load_mwmed = bc[0]
#lambda_ = bc[1]
df.load_mwmed = np.log(df.load_mwmed) # série logaritmizada


# split treino-teste
n_test = 31
train, test = train_test_split(df, n_test)


# auto-arima 
# (Best model log:  ARIMA(1,1,2)(1,0,1)[7])
SARIMA_model = pm.auto_arima(train, 
                            start_p=1, 
                            start_q=1,
                            test='kpss', 
                            max_p=4, max_q=5, # maximum p and q
                            max_P=2, max_Q=2,
                            #d=None,
                            max_d = 2,
                            #D=None# let model determine 'd'
                            max_D = 2,
                            seasonal=True, 
                            m=7, # frequency of series (if m=1, seasonal is set to FALSE automatically)
                            trace=True, #logs 
                            error_action='warn', #shows errors ('ignore' silences these)
                            suppress_warnings=True,
                            stepwise=False,
                            information_criterion='bic')

#model1 = sm.tsa.statespace.SARIMAX(train,order=(1, 1, 2),seasonal_order=(1,0,1,7), trend='c')
#SARIMA_model = model1.fit()
#load_fc = SARIMA_model.forecast(n_test)

# parâmetros
print(SARIMA_model.summary())

# diagnósticos dos resíduos
SARIMA_model.plot_diagnostics(figsize=(15,5))
plt.subplots_adjust(hspace= 0.7)
plt.show()

# forecast
load_fc = pd.Series(SARIMA_model.predict(n_periods=n_test)) # transforma forecast em Series
load_fc.index = test.index # deixa o forecast e o teste com os mesmo índices para plotar



# medidas de acurácia
#fc = inv_boxcox(fc, lambda_) # volta a escala para o original
#test = inv_boxcox(test, lambda_)
load_fc = np.exp(load_fc) # volta a escala para o original
load_test = np.exp(test)
medidas_fc = get_measures(load_fc, load_test) 
df_medidas_fc = pd.DataFrame([medidas_fc])
print(df_medidas_fc)

# visualização do forecast
plt.figure(figsize = (15, 5))
plt.plot(load_fc.values, c = "red", label = "forecast")
plt.plot(load_test.values, c = "blue", label = "actual")
plt.legend()
plt.show()

# write forecst csv
load_fc.index.names = ["date"]
load_fc.index = test.index 
load_fc.columns = ["forecast"]
load_fc.to_csv("validation/auto_arima_fc.csv")
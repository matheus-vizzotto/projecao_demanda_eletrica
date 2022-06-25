# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 00:20:41 2022

@author: Matheus
"""

import pandas as pd
#import numpy as np
from load import load_data
from scipy.special import boxcox, inv_boxcox
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
df = boxcox(df, 2.5)

# split treino-teste
train, test = df.iloc[:-30], df.iloc[-30:] 

# auto-arima (Best model boxcox:  ARIMA(4,1,2)(1,0,1)[7] intercept)
SARIMA_model = pm.auto_arima(train, 
                            start_p=1, 
                            start_q=1,
                            test='kpss', # use adftest to find optimal 'd'
                            max_p=4, max_q=5, # maximum p and q
                            max_P=1, max_Q=1,
                            m=7, # frequency of series (if m==1, seasonal is set to FALSE automatically)
                            #d=None,
                            max_d = 2,
                            #D=None# let model determine 'd'
                            max_D = 1,
                            seasonal=True, # No Seasonality for standard ARIMA
                            trace=True, #logs 
                            error_action='warn', #shows errors ('ignore' silences these)
                            suppress_warnings=True,
                            stepwise=True,
                            information_criterion='bic')

# diagnósticos dos resíduos
SARIMA_model.plot_diagnostics(figsize=(15,5))
plt.subplots_adjust(hspace= 0.7)
plt.show()

# forecast
fc = pd.Series(SARIMA_model.predict(n_periods=30)) # transforma forecast em Series
fc = inv_boxcox(fc, 2.5) # volta a escala para o original
fc.index = test.index # deixa o forecast e o teste com os mesmo índices para plotar
medidas_fc = get_measures(fc, test) # medidas de acurácia
#print("Medidas de acurácia:\n", medidas_fc)
print(pd.DataFrame(medidas_fc))

plt.figure(figsize = (15, 5))
plt.plot(fc, c = "red", label = "forecast")
plt.plot(test, c = "blue", label = "actual")
plt.legend()
plt.show()



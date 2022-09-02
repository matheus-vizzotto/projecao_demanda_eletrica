import pandas as pd
import numpy as np
from load import load_data
from load import train_test_split
from load import get_measures
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import statsmodels.api as sm
import warnings 

# configs
warnings.filterwarnings('ignore') # remove warnings
plt.style.use('fivethirtyeight')
rcParams['figure.figsize'] = 15, 5

# load data
df = load_data()
sns.heatmap(df.isna().transpose()) # missing values

# visualize data
df.plot(title = "Série em nível")
df["load_mwmed"].interpolate(method = "linear", inplace = True)  # fill empty values
df.load_mwmed = np.log(df.load_mwmed) # log-transform
df.plot(title = "Série transformada")

# split train - test
n_test = 10
train, test = train_test_split(df, n_test)

# fit model
model1 = sm.tsa.statespace.SARIMAX(train,order=(1, 1, 2),seasonal_order=(1,0,1,7), trend='c')
SARIMA_model = model1.fit() #method='cg'

# parameters
print(SARIMA_model.summary())

# residuals diagnostics
SARIMA_model.plot_diagnostics(figsize=(15,5))
plt.subplots_adjust(hspace= 0.7)
plt.show()

# forecast
load_fc = SARIMA_model.forecast(n_test)
load_fc.index = test.index # deixa o forecast e o teste com os mesmo índices para plotar


# model's accuracy
load_fc = np.exp(load_fc) 
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
load_fc.to_csv("validation/arima_fc.csv")
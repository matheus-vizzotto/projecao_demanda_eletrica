import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing, Holt
from load import load_data
from load import train_test_split
from load import get_measures


import warnings # retirar avisos
warnings.filterwarnings('ignore')
rcParams['figure.figsize'] = 15, 5

df = load_data()
df["load_mwmed"].interpolate(method = "linear", inplace = True)


n_test = 15
train, test = train_test_split(df, n_test)

fit1 = ExponentialSmoothing(train ,seasonal_periods=7,trend='add', seasonal='add',).fit()
y_hat = fit1.forecast(n_test)

plt.plot(y_hat.reset_index(drop = True), label = "forecast")
plt.plot(test.load_mwmed.reset_index(drop = True), label = "test")
plt.legend()
plt.show()


y_hat.index.names = ["date"]
y_hat.index = test.index 
y_hat.columns = ["forecast"]
y_hat.to_csv("validation/hw_fc.csv")


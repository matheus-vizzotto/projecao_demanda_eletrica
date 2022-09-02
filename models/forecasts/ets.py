import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams
from sktime.forecasting.ets import AutoETS
from load import load_data
from load import train_test_split
from load import get_measures


import warnings # retirar avisos
warnings.filterwarnings('ignore')
rcParams['figure.figsize'] = 15, 5

df = load_data()
df["load_mwmed"].interpolate(method = "linear", inplace = True)


n_test = 10
train, test = train_test_split(df, n_test)

forecaster = AutoETS(auto=True, n_jobs=-1, seasonal = "add", sp=7)
forecaster.fit(train.load_mwmed.values)

y_pred = forecaster.predict(fh=[x for x in range(0, n_test)])
y_pred = [x for x in y_pred.flatten()]

medidas_fc = get_measures(pd.Series(y_pred), test.load_mwmed) 
df_medidas_fc = pd.DataFrame([medidas_fc])
print(df_medidas_fc)


plt.plot(y_pred, label = "forecast")
plt.plot(test.load_mwmed.reset_index(drop = True), label = "test")
plt.legend()
plt.show()

y_pred = pd.Series(y_pred)
y_pred.index.names = ["date"]
y_pred.index = test.index 
y_pred.columns = ["forecast"]
y_pred.to_csv("validation/ets_fc.csv")


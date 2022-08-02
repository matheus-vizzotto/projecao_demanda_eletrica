# -*- coding: utf-8 -*-
import requests
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from math import sqrt


def download_carga_horaria(ano_inicio: int, ano_fim: int):
    """ Função para fazer download dos dados de carga elétrica por subsistema no período de referência em base diária."""

    #url = "https://ons-dl-prod-opendata.s3.amazonaws.com/dataset/carga_energia_di/CARGA_ENERGIA_{}.csv"
    url = "https://ons-dl-prod-opendata.s3.amazonaws.com/dataset/curva-carga-ho/CURVA_CARGA_{}.csv"
    # verificar se anos inicial e final estão disponíveis
    get0 = requests.get(url.format(ano_inicio)).status_code # verify = False (autenticação)
    getn = requests.get(url.format(ano_fim)).status_code 
    if (get0 == 200) and (getn == 200): # 200: página (ano) disponível

        # concatenar arquivos de cada ano em um único dataframe
        df = pd.DataFrame()
        for ano in range(ano_inicio, ano_fim + 1):
            df2 = pd.read_csv(url.format(ano), sep = ";")
            df = pd.concat([df, df2])
        df.columns = ["id_reg", "desc_reg", "date", "load_mwmed"]
        df.loc[:, "date"] = pd.to_datetime(df.loc[:, "date"], format = '%Y-%m-%d %H:%M:%S')
        df.sort_values(by = "date", inplace = True)
        return df
    
    else:
       print("Ano não disponível.")


def load_data():
    """
    Função para ler e transformar os dados já presentes no diretório especificado
    """
    path = "../data/hourly_load.csv"
    df_load = pd.read_csv(path, parse_dates = ["date"])
    df_load2 = df_load[df_load["id_reg"] == "S"]           # região sul
    df_load3 = df_load2[df_load2["date"] <= '2022-05-31']  # data de corte
    df_load4 = df_load3[["date", "load_mwmed"]].set_index("date")
    return df_load4

def get_measures(forecast, test):
    """
    Função para obter medidas de acurária a partir dos dados de projeção e teste
    """
    forecast.reset_index(drop = True, inplace = True)
    test.reset_index(drop = True, inplace = True)
    if isinstance(forecast, pd.Series) and isinstance(test, pd.Series):
        errors = [(test.iloc[i] - forecast.iloc[i])**2 for i in range(len(test))]
    else:
        errors = [(test.iloc[i][0] - forecast.iloc[i])**2 for i in range(len(test))]
    mae = mean_absolute_error(test, forecast)
    mse = mean_squared_error(test, forecast)
    rmse = sqrt(mse)
    mape = mean_absolute_percentage_error(test, forecast)
    # smape
    a = np.reshape(test.values, (-1,))
    b = np.reshape(forecast.values, (-1,))
    smape = np.mean(100*2.0 * np.abs(a - b) / (np.abs(a) + np.abs(b))).item()
    # dicionário com as medidas de erro
    measures = { "erro": sum(errors),
                 "mae": mae,
                 "mse": mse,
                 "rmse": rmse,
                 "mape": mape,
                 "smape": smape
                }
    # arredondamento
    for key, item in measures.items():
        measures[key] = round(measures[key], 4)
    return measures

def series_to_supervised(data, n_in = 1, n_out = 1, dropnan = True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis = 1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace = True)
    return agg

def train_test_split(data, n_test):
    if isinstance(data, pd.DataFrame):
        train, test = data.iloc[:-n_test, :], data.iloc[-n_test:, :]
    elif isinstance(data, np.ndarray):
        train, test = data[:-n_test, :], data[-n_test:, :]
    return train, test
    
def create_features(df, datetime_column):
    """ Função para criar as variáveis de calendário com base na coluna de data selecionada. """
    x = df[datetime_column]
    df["ano"] = x.dt.year
    df["trimestre"] = x.dt.quarter         
    df["mes"] = x.dt.month
    df["dia"] = x.dt.day
    df["dia_ano"] = x.dt.dayofyear
    df["dia_semana"] = x.dt.weekday + 1    # 0: segunda-feira; 6: domingo
    df["semana_ano"] = x.dt.isocalendar().week
    df["apagao"] = x.dt.year.apply(lambda x: 1 if x in [2001, 2002] else 0)
    df['Series'] = [i for i in range(1, len(x) + 1)]
    
    return df

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
# -*- coding: utf-8 -*-
import requests
import pandas as pd


def download_carga(ano_inicio: int, ano_fim: int):
    """ Função para fazer download dos dados de carga elétrica por subsistema no período de referência em base diária."""

    url = "https://ons-dl-prod-opendata.s3.amazonaws.com/dataset/carga_energia_di/CARGA_ENERGIA_{}.csv"

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
        df.loc[:, "date"] = pd.to_datetime(df.loc[:, "date"], format = '%Y-%m-%d')
        df.sort_values(by = "date", inplace = True)
        return df
    
    else:
       print("Ano não disponível.")
       
       
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

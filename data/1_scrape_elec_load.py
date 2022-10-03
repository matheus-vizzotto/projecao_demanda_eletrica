import pandas as pd

ano_inicio = 2000
ano_fim = 2022

url = "https://ons-dl-prod-opendata.s3.amazonaws.com/dataset/carga_energia_di/CARGA_ENERGIA_{}.csv"
df = pd.DataFrame()
for ano in range(ano_inicio, ano_fim + 1):
    print(f"Extraindo ano {ano}...")
    df2 = pd.read_csv(url.format(ano), sep = ";")
    df = pd.concat([df, df2])
    print(f"    Concluído.")

df.columns = ["id_reg", "desc_reg", "date", "load_mwmed"]
df.loc[:, "date"] = pd.to_datetime(df.loc[:, "date"], format = '%Y-%m-%d')
df.sort_values(by = "date", inplace = True)
df = df.iloc[ :-4 , : ] # t - 1 fica com NaN

df.to_csv("daily_load.csv", index = False)

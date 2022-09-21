import pandas as pd
import os
import json

ensemble = ["hw", "prophet"]
folds = 15
cvs = [f"cv_{x}" for x in range(1,folds+1)]


df_preds = pd.DataFrame()
for file in os.listdir("validation"):
    if file[:-8] in ensemble:
        file_path = "validation/" + file
        with open(file_path) as f:
            fc_dict = json.load(f)
        preds = []
        for cv in cvs:
            preds_cv = fc_dict[cv]["pred"]
            preds.append(pd.Series(preds_cv))
        df_preds_model = pd.concat(preds, axis = 0)
        df_preds = pd.concat([df_preds, df_preds_model], axis = 1)
print(df_preds)

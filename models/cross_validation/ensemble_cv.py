import pandas as pd
import os
import json
import numpy as np
from collections import defaultdict

ensemble = ["hw", "prophet"]
folds = 15
cvs = [f"cv_{x}" for x in range(folds,0, -1)]

# df_preds = pd.DataFrame()
# for file in os.listdir("validation"):
#     if file[:-8] in ensemble:
#         file_path = "validation/" + file
#         with open(file_path) as f:
#             fc_dict = json.load(f)
#         preds = []
#         for cv in cvs:
#             preds_cv = fc_dict[cv]["pred"]
#             preds.append(pd.Series(preds_cv))
#         df_preds_model = pd.concat(preds, axis = 0)
#         df_preds = pd.concat([df_preds, df_preds_model], axis = 1)
# df_preds = df_preds.mean(axis=1)
# print(df_preds)

with open("validation/hw_cv.json") as f:
    hw_dict = json.load(f)
with open("validation/prophet_cv.json") as f:
    prophet_dict = json.load(f)

ensemble_dict = defaultdict(dict)
l = pd.DataFrame()
for cv in cvs:
    hw_pred = hw_dict[cv]["pred"]
    prophet_pred = prophet_dict[cv]["pred"]
    test = hw_dict[cv]["test"]
    lz = list(zip(hw_pred, prophet_pred))
    ensemble = [np.mean(x) for x in lz]
    ensemble_dict[f"cv_{cv}"]["pred"] = ensemble
    ensemble_dict[f"cv_{cv}"]["test"] = test
    ensemble_s = pd.Series(ensemble)
    l = pd.concat([l, ensemble_s], axis = 0)
d = dict(ensemble_dict)

with open('validation/ensemble_cv.json', 'w') as f:
    json.dump(d, f)
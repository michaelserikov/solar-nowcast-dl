import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from snc.data import load_grid, clear_sky_proxy
from snc.baselines import *

idx, field, la, lo = load_grid()
train = idx.year <= 2023
cs = clear_sky_proxy(idx[train], field[train])           # per month-hour from training years
keys = idx.month.values * 24 + idx.hour.values
tk = idx[train].month.values * 24 + idx[train].hour.values
csfull = np.zeros_like(field)
for k in np.unique(keys):
    csfull[keys == k] = np.quantile(field[train][tk == k], 0.98, axis=0)
kt = kt_field(field, csfull)
test = idx.year == 2025
res = {}
for h in HORIZONS:
    t = np.arange(2, len(idx) - h)
    t = t[test[t + h]]
    true = field[t + h]; cst = csfull[t + h]
    clim = cst * np.nanmean(kt[train & (csfull.max(axis=(1, 2)) > 0)], axis=0)  # mean kt per cell
    res[h] = {
        "persistence_ghi": rmse_daytime(field[t], true, cst),
        "smart_persistence": rmse_daytime(smart_persistence(kt[t], cst), true, cst),
        "climatological_kt": rmse_daytime(clim * np.ones_like(true), true, cst),
        "n_daytime_cells": int((cst > CS_MIN).sum()),
    }
    print(h, {k: round(v, 1) for k, v in res[h].items()})
Path("results").mkdir(exist_ok=True)
json.dump(res, open("results/baselines_2025.json", "w"), indent=1)

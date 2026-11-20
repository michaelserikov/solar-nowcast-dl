from __future__ import annotations
import glob, json
from pathlib import Path
import numpy as np, pandas as pd

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
NLAT = NLON = 10


def load_grid() -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray]:
    """Returns time index, field [T, 10, 10] in W/m2, latitudes, longitudes."""
    series, lat, lon = {}, {}, {}
    for f in sorted(glob.glob(str(RAW / "*.json"))):
        year = int(Path(f).name[:4]); blk = int(Path(f).name[5:8])
        d = json.load(open(f, encoding="utf-8")); d = d if isinstance(d, list) else [d]
        for i, p in enumerate(d):
            k = blk + i
            series.setdefault(k, []).append(np.array(p["hourly"]["shortwave_radiation"], dtype="float32"))
            lat[k], lon[k] = p["latitude"], p["longitude"]
    K = sorted(series)
    arr = np.stack([np.concatenate(series[k]) for k in K], axis=1)      # [T, 100]
    idx = pd.date_range("2022-01-01", periods=arr.shape[0], freq="h", tz="UTC")
    field = arr.reshape(-1, NLAT, NLON)
    return idx, field, np.array([lat[k] for k in K]).reshape(NLAT, NLON), np.array([lon[k] for k in K]).reshape(NLAT, NLON)


def clear_sky_proxy(idx: pd.DatetimeIndex, field: np.ndarray, q: float = 0.98) -> np.ndarray:
    """Clear-sky radiation proxy per grid cell, hour of day and month: the q-quantile of observed
    radiation in that cell and hour-month over the training years (a cheap stand-in for a physical model).
    Returns an array shaped like field."""
    keys = idx.month.values * 24 + idx.hour.values
    out = np.zeros_like(field)
    for k in np.unique(keys):
        m = keys == k
        out[m] = np.quantile(field[m], q, axis=0)
    return out

"""Satellite-derived hourly shortwave radiation on a regular grid over Germany (Open-Meteo satellite archive,
DWD/EUMETSAT-based products). Cached per year and block of points; resumable."""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, requests

OUT = Path(__file__).resolve().parents[1] / "data" / "raw"; OUT.mkdir(parents=True, exist_ok=True)
LATS = np.round(np.linspace(47.5, 54.5, 10), 3)
LONS = np.round(np.linspace(6.0, 14.5, 10), 3)
URL = "https://satellite-api.open-meteo.com/v1/archive"
pts = [(a, o) for a in LATS for o in LONS]
for year in (2022, 2023, 2024, 2025):
    for b in range(0, len(pts), 10):
        f = OUT / f"{year}_{b:03d}.json"
        if f.exists() and f.stat().st_size > 1000:
            continue
        blk = pts[b:b + 10]
        q = dict(latitude=",".join(str(p[0]) for p in blk), longitude=",".join(str(p[1]) for p in blk),
                 start_date=f"{year}-01-01", end_date=f"{year}-12-31", hourly="shortwave_radiation", timezone="UTC")
        for attempt in range(6):
            r = requests.get(URL, params=q, timeout=120)
            if r.status_code == 200:
                f.write_text(r.text, encoding="utf-8"); print("ok", f.name, flush=True); break
            print("retry", r.status_code, r.text[:120], flush=True); time.sleep(30 * (attempt + 1))
        time.sleep(2)
print("done")

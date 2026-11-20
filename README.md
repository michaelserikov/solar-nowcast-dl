# Nowcasting satellite-derived solar irradiance over Germany with a convolutional network

Forecasts of hourly shortwave radiation 1 to 6 hours ahead on a 10 x 10 grid over Germany (47.5-54.5 N, 6-14.5 E), compared with persistence and ridge regression baselines. Inputs are the last three hours of the clear-sky index on the whole grid plus the clear-sky level and time of day at the target hour.

## Data

Open-Meteo satellite archive (hourly shortwave radiation from satellite-based products, no account needed), 100 grid points, 2022-2025 (35,064 hours, no gaps). Training 2022-2023, validation 2024 (early stopping), test 2025. Clear-sky level: 98th percentile of the training-year radiation per cell, month and hour (a proxy, not a physical clear-sky model). Scores are RMSE in W/m2 over daytime cells (clear-sky level above 50 W/m2; 415,374 cell-hours in 2025).

## Results (test year 2025, RMSE W/m2)

| Horizon | Smart persistence | Ridge, 5x5 neighbourhood | CNN |
|---|---|---|---|
| 1 h | 67.1 | 62.0 | 53.6 |
| 2 h | 96.8 | 80.7 | 67.0 |
| 3 h | 131.0 | 95.0 | 79.8 |
| 4 h | 170.8 | 106.8 | 90.7 |
| 6 h | 251.7 | 121.3 | 108.6 |

Climatological clear-sky index: 161 at every horizon. The CNN improves on smart persistence by 20% at 1 h and 57% at 6 h, and on ridge by 14-17% (`results/models_2025.json`, `scripts/run_models.py`, `scripts/run_baselines.py`).

## Probabilistic version (v0.2)

A convolutional network that outputs the 10%, 50% and 90% quantiles of the clear-sky index (pinball loss, non-crossing by construction), against two reference methods that use the same residual quantiles taken from the training years: smart persistence and an advected field (global integer shift between the last two hours, repeated for the horizon). Test year 2025, daytime cells (`scripts/run_prob.py`, `results/probabilistic_2025.json`).

| Horizon | Pinball loss, persistence | Pinball loss, advection | Pinball loss, quantile CNN | Coverage of the 80% interval (CNN) | Median RMSE W/m2 (CNN) |
|---|---|---|---|---|---|
| 1 h | 15.0 | 14.7 | 10.0 | 0.78 | 53.4 |
| 2 h | 22.9 | 22.4 | 13.2 | 0.79 | 67.2 |
| 4 h | 34.0 | 33.7 | 18.3 | 0.79 | 90.1 |
| 6 h | 40.5 | 40.7 | 22.1 | 0.82 | 110.8 |

The CNN lowers the pinball loss by 33% (1 h) to 46% (6 h) against both reference methods and its intervals hold about 80% (0.78 to 0.82). Advection barely beats persistence: with a grid spacing of about 80 km and hourly steps, clouds move less than a cell per hour, so a global shift rarely changes the field. This is a limit of the data (hourly point products), not a statement about advection on satellite images at 5 to 15 minutes.

## Limits

- **Hourly point products on a grid, not satellite images.** At hourly steps and a grid spacing of about 80 km, cloud motion is resolved only coarsely; this is not a 5-15 minute image nowcast and the results do not transfer to one.
- One training run per horizon and one test year; no confidence intervals. The 2-hour model stopped at the 40-epoch cap. Ridge sees a 5 x 5 neighbourhood, the CNN a wider field through dilation, so the gap mixes model class and context.
- The clear-sky proxy is empirical; evaluation is on radiation, not on PV power; no cloud-type or aerosol input, no numerical weather forecast input.
- Research study on public data.

## Run

```bash
python scripts/download_grid.py && python scripts/run_baselines.py && python scripts/run_models.py   # needs torch (CPU is enough)
```

## Data licence

Radiation data: Open-Meteo satellite archive (https://open-meteo.com), CC BY 4.0, derived from satellite-based radiation products. The raw downloads are included in `data/raw/`.

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

## Limits

- **Hourly point products on a grid, not satellite images.** At hourly steps and a grid spacing of about 80 km, cloud motion is resolved only coarsely; this is not a 5-15 minute image nowcast and the results do not transfer to one.
- One training run per horizon and one test year; no confidence intervals. The 2-hour model stopped at the 40-epoch cap. Ridge sees a 5 x 5 neighbourhood, the CNN a wider field through dilation, so the gap mixes model class and context.
- The clear-sky proxy is empirical; evaluation is on radiation, not on PV power; no cloud-type or aerosol input, no numerical weather forecast input.
- Research study on public data.

## Run

```bash
python scripts/download_grid.py && python scripts/run_baselines.py && python scripts/run_models.py   # needs torch (CPU is enough)
```

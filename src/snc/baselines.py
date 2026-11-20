from __future__ import annotations
import numpy as np

HORIZONS = (1, 2, 3, 4, 6)
CS_MIN = 50.0   # W/m2: daytime cells only


def kt_field(field, cs):
    return np.clip(field / np.maximum(cs, CS_MIN), 0.0, 1.3)


def samples(field, cs, hours_utc, horizon, lags=3):
    """Index set of forecast origins t with all lags and the target t+h available; returns t array."""
    T = field.shape[0]
    return np.arange(lags - 1, T - horizon)


def smart_persistence(kt_now, cs_target):
    return kt_now * cs_target


def rmse_daytime(pred, true, cs_target):
    m = cs_target > CS_MIN
    return float(np.sqrt(np.mean((pred[m] - true[m]) ** 2)))


def mae_daytime(pred, true, cs_target):
    m = cs_target > CS_MIN
    return float(np.mean(np.abs(pred[m] - true[m])))

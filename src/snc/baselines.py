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


def advect(kt_now, kt_prev, h):
    """Shift the field by the global integer motion (cells per hour) that best maps kt(t-1) onto kt(t), repeated h times."""
    best, bs = None, (0, 0)
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            sh = np.roll(np.roll(kt_prev, dy, axis=0), dx, axis=1)
            sl = (slice(max(dy, 0) + 0, 10 + min(dy, 0)), slice(max(dx, 0) + 0, 10 + min(dx, 0)))
            err = np.mean((sh[sl] - kt_now[sl]) ** 2)
            if best is None or err < best:
                best, bs = err, (dy, dx)
    out = np.roll(np.roll(kt_now, bs[0] * h, axis=0), bs[1] * h, axis=1)
    # cells that wrapped around are replaced by the unshifted field
    if bs[0] * h > 0: out[:min(bs[0] * h, 10)] = kt_now[:min(bs[0] * h, 10)]
    if bs[0] * h < 0: out[max(10 + bs[0] * h, 0):] = kt_now[max(10 + bs[0] * h, 0):]
    if bs[1] * h > 0: out[:, :min(bs[1] * h, 10)] = kt_now[:, :min(bs[1] * h, 10)]
    if bs[1] * h < 0: out[:, max(10 + bs[1] * h, 0):] = kt_now[:, max(10 + bs[1] * h, 0):]
    return out

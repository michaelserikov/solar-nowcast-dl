from __future__ import annotations
import numpy as np
from .baselines import CS_MIN


def clear_sky(field, idx, train_mask):
    keys = idx.month.values * 24 + idx.hour.values
    out = np.zeros_like(field)
    for k in np.unique(keys):
        m = keys == k
        if not (train_mask & m).any():   # no training data for this month and hour: leave 0 (no forecast)
            continue
        out[m] = np.quantile(field[train_mask & m], 0.98, axis=0)
    return out


def kt_of(field, cs):
    return np.clip(field / np.maximum(cs, CS_MIN), 0.0, 1.3).astype("float32")


def time_feats(idx):
    h = idx.hour.values / 24.0 * 2 * np.pi
    d = (idx.dayofyear.values / 365.25) * 2 * np.pi
    return np.stack([np.sin(h), np.cos(h), np.sin(d), np.cos(d)], axis=1).astype("float32")


def make_inputs(kt, cs, tf, t, h, lags=3):
    """Grid input for origins t and horizon h: channels = kt at t, t-1, t-2, cs(t+h)/1000, then time features."""
    chans = [kt[t - l] for l in range(lags)]
    chans.append(cs[t + h] / 1000.0)
    x = np.stack(chans, axis=1)                                   # [N, C, 10, 10]
    tfe = np.broadcast_to(tf[t + h][:, :, None, None], (len(t), tf.shape[1], 10, 10))
    return np.concatenate([x, tfe], axis=1).astype("float32")

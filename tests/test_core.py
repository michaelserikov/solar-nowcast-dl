import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from snc.baselines import rmse_daytime, smart_persistence, CS_MIN, advect
from snc.features import clear_sky, kt_of, make_inputs, time_feats


def synthetic(T=24 * 60):
    idx = pd.date_range("2022-01-01", periods=T, freq="h", tz="UTC")
    day = np.clip(np.sin((idx.hour.values - 6) / 12 * np.pi), 0, None)[:, None, None] * 800
    return idx, (day * np.ones((T, 10, 10))).astype("float32")


def test_clear_sky_day_has_kt_one():
    idx, f = synthetic()
    cs = clear_sky(f, idx, np.ones(len(idx), bool))
    kt = kt_of(f, cs)
    assert np.allclose(kt[f > CS_MIN], 1.0, atol=1e-3)


def test_smart_persistence_perfect_on_clear_sky():
    idx, f = synthetic()
    cs = clear_sky(f, idx, np.ones(len(idx), bool)); kt = kt_of(f, cs)
    t = np.arange(2, len(idx) - 3)
    t = t[f[t].min(axis=(1, 2)) > CS_MIN]          # kt is undefined at night, so origins in daylight only
    assert rmse_daytime(smart_persistence(kt[t], cs[t + 3]), f[t + 3], cs[t + 3]) < 1.0


def test_no_leak_clear_sky_uses_train_only():
    idx, f = synthetic()
    f2 = f.copy(); f2[-24 * 5:] *= 5.0                      # change the last days only
    tr = np.arange(len(idx)) < len(idx) - 24 * 5
    assert np.allclose(clear_sky(f, idx, tr), clear_sky(f2, idx, tr))


def test_input_shape_and_no_future():
    idx, f = synthetic(); cs = clear_sky(f, idx, np.ones(len(idx), bool)); kt = kt_of(f, cs)
    t = np.array([10, 11]); x = make_inputs(kt, cs, time_feats(idx), t, 3)
    assert x.shape == (2, 3 + 1 + 4, 10, 10)
    kt2 = kt.copy(); kt2[12:] = 9.0                         # values after origin 11 must not enter
    assert np.allclose(make_inputs(kt2, cs, time_feats(idx), t, 3)[:, :3], x[:, :3])


def test_advection_recovers_a_shifted_pattern():
    rng = np.random.default_rng(0)
    prev = rng.random((10, 10)).astype("float32")
    now = np.roll(prev, 1, axis=1)                  # pattern moved one cell east
    out = advect(now, prev, 1)
    assert np.allclose(out[:, 2:], np.roll(now, 1, axis=1)[:, 2:])   # continues one more cell
    assert np.allclose(advect(now, now, 3), now)    # no motion: unchanged

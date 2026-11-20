"""Ridge on 5x5 neighbourhoods and a small CNN against persistence baselines; train 2022-2023, validate 2024, test 2025."""
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from snc.data import load_grid
from snc.features import *
from snc.baselines import HORIZONS, CS_MIN

try:
    import torch, torch.nn as nn
except ImportError:
    torch = None
torch and torch.manual_seed(0); np.random.seed(0)

idx, field, la, lo = load_grid()
tr = idx.year <= 2023; va = idx.year == 2024; te = idx.year == 2025
cs = clear_sky(field, idx, tr)
kt = kt_of(field, cs); tf = time_feats(idx)
H = list(HORIZONS)
rmse = lambda p, y, c: float(np.sqrt(np.mean((p[c > CS_MIN] - y[c > CS_MIN]) ** 2)))


def origins(mask, h):
    t = np.arange(2, len(idx) - h); return t[mask[t + h]]


class Net(nn.Module):
    def __init__(self, cin, nh):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(cin, 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, 32, 3, padding=1), nn.GELU(),
                               nn.Conv2d(32, 32, 3, padding=2, dilation=2), nn.GELU(), nn.Conv2d(32, 1, 1))
    def forward(self, x):
        return self.f(x)[:, 0]

results = {h: {} for h in H}
for h in H:
    ttr, tva, tte = origins(tr, h), origins(va, h), origins(te, h)
    # baselines on test
    true = field[tte + h]; cst = cs[tte + h]
    results[h]["smart_persistence"] = rmse(kt[tte] * cst, true, cst)
    # ridge on 5x5 neighbourhood
    from sklearn.linear_model import Ridge
    def patch_feats(t):
        k = np.pad(np.stack([kt[t - l] for l in range(3)], 1), ((0, 0), (0, 0), (2, 2), (2, 2)), mode="edge")
        P = np.stack([k[:, :, i:i + 10, j:j + 10] for i in range(5) for j in range(5)], axis=2)   # [N,3,25,10,10]
        P = P.reshape(len(t), 75, 10, 10).transpose(0, 2, 3, 1).reshape(-1, 75)
        extra = np.concatenate([(cs[t + h] / 1000.0).reshape(-1, 1), np.repeat(tf[t + h], 100, axis=0)], axis=1)
        return np.concatenate([P, extra], axis=1).astype("float32")
    def fit_rows(t, step=1):
        t = t[::step]; X = patch_feats(t); y = kt[t + h].reshape(-1); c = cs[t + h].reshape(-1)
        m = c > CS_MIN; return X[m], y[m]
    X, y = fit_rows(ttr, 2)
    rg = Ridge(alpha=10.0).fit(X, y)
    Xt = patch_feats(tte); pk = np.clip(rg.predict(Xt), 0, 1.3).reshape(len(tte), 10, 10)
    results[h]["ridge_5x5"] = rmse(pk * cst, true, cst)
    # CNN
    if torch is not None:
        Xtr, Xva, Xte = [make_inputs(kt, cs, tf, t, h) for t in (ttr, tva, tte)]
        Ytr, Yva = kt[ttr + h], kt[tva + h]
        Wtr, Wva = (cs[ttr + h] > CS_MIN).astype("float32"), (cs[tva + h] > CS_MIN).astype("float32")
        net = Net(Xtr.shape[1], 1); opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
        best, bst, patience = 1e9, None, 0
        Xtr_t, Ytr_t, Wtr_t = map(torch.from_numpy, (Xtr, Ytr, Wtr))
        Xva_t, Yva_t, Wva_t = map(torch.from_numpy, (Xva, Yva, Wva))
        for ep in range(40):
            net.train(); perm = torch.randperm(len(Xtr_t))
            for i in range(0, len(perm), 256):
                b = perm[i:i + 256]
                loss = (((net(Xtr_t[b]) - Ytr_t[b]) ** 2) * Wtr_t[b]).sum() / Wtr_t[b].sum().clamp(min=1)
                opt.zero_grad(); loss.backward(); opt.step()
            net.eval()
            with torch.no_grad():
                vl = float((((net(Xva_t) - Yva_t) ** 2) * Wva_t).sum() / Wva_t.sum())
            if vl < best - 1e-5: best, bst, patience = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
            else:
                patience += 1
                if patience >= 5: break
        net.load_state_dict(bst); net.eval()
        with torch.no_grad(): pc = net(torch.from_numpy(Xte)).numpy().clip(0, 1.3)
        results[h]["cnn"] = rmse(pc * cst, true, cst)
        results[h]["cnn_epochs"] = ep + 1
    print(h, {k: round(v, 1) for k, v in results[h].items()}, flush=True)
Path("results").mkdir(exist_ok=True)
json.dump(results, open("results/models_2025.json", "w"), indent=1)

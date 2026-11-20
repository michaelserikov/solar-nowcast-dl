"""Probabilistic nowcast (quantile CNN) against an advection baseline and a persistence-plus-residual-quantiles baseline.

Train 2022-2023, validate 2024, test 2025. Quantiles of the clear-sky index at 0.1, 0.5, 0.9 (pinball loss).
Scores on daytime cells in W/m2: median RMSE, mean pinball loss over the three quantiles, coverage of the 80% interval.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from snc.baselines import CS_MIN, advect
from snc.data import load_grid
from snc.features import clear_sky, kt_of, make_inputs, time_feats

torch.manual_seed(0)
np.random.seed(0)
QS = np.array([0.1, 0.5, 0.9], dtype="float32")
idx, field, la, lo = load_grid()
tr, va, te = idx.year <= 2023, idx.year == 2024, idx.year == 2025
cs = clear_sky(field, idx, tr)
kt = kt_of(field, cs)
tf = time_feats(idx)


def origins(mask, h):
    t = np.arange(2, len(idx) - h)
    return t[mask[t + h]]


def pinball(pred, y, w):          # pred [N,3,10,10] in kt, y [N,10,10]
    q = torch.tensor(QS).view(1, 3, 1, 1)
    e = y.unsqueeze(1) - pred
    l = torch.maximum(q * e, (q - 1) * e)
    return (l * w.unsqueeze(1)).sum() / (w.sum() * 3).clamp(min=1)


def score(q_kt, true, cst):
    """q_kt [N,3,10,10] kt quantiles -> W/m2 scores on daytime cells."""
    m = cst > CS_MIN
    qw = q_kt * cst[:, None]
    med = qw[:, 1][m]
    y = true[m]
    rmse = float(np.sqrt(np.mean((med - y) ** 2)))
    pb = []
    for i, q in enumerate(QS):
        e = y - qw[:, i][m]
        pb.append(np.mean(np.maximum(q * e, (q - 1) * e)))
    cover = float(np.mean((y >= qw[:, 0][m]) & (y <= qw[:, 2][m])))
    return {"median_rmse": rmse, "pinball": float(np.mean(pb)), "coverage_80": cover}


class Net(nn.Module):
    def __init__(self, cin):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(cin, 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, 32, 3, padding=1), nn.GELU(),
                               nn.Conv2d(32, 32, 3, padding=2, dilation=2), nn.GELU(), nn.Conv2d(32, 3, 1))

    def forward(self, x):
        o = self.f(x)
        base = o[:, 1:2]                                  # sorted quantiles: median plus positive gaps
        lo = base - torch.nn.functional.softplus(o[:, 0:1])
        hi = base + torch.nn.functional.softplus(o[:, 2:3])
        return torch.cat([lo, base, hi], dim=1)


res = {}
for h in (1, 2, 4, 6):
    ttr, tva, tte = origins(tr, h), origins(va, h), origins(te, h)
    true, cst = field[tte + h], cs[tte + h]
    # baseline 1: smart persistence with empirical residual quantiles from the training years
    pers_tr = kt[ttr]
    resid = (kt[ttr + h] - pers_tr)[cs[ttr + h] > CS_MIN]
    rq = np.quantile(resid, QS)
    b1 = np.stack([np.clip(kt[tte] + r, 0, 1.3) for r in rq], axis=1)
    # baseline 2: advected field with the same residual quantiles (residuals from training years)
    adv_tr = np.stack([advect(kt[t], kt[t - 1], h) for t in ttr[::6]])
    resid2 = (kt[ttr[::6] + h] - adv_tr)[cs[ttr[::6] + h] > CS_MIN]
    rq2 = np.quantile(resid2, QS)
    adv_te = np.stack([advect(kt[t], kt[t - 1], h) for t in tte])
    b2 = np.stack([np.clip(adv_te + r, 0, 1.3) for r in rq2], axis=1)
    # quantile CNN
    Xtr, Xva, Xte = [make_inputs(kt, cs, tf, t, h) for t in (ttr, tva, tte)]
    Ytr, Yva = kt[ttr + h], kt[tva + h]
    Wtr, Wva = (cs[ttr + h] > CS_MIN).astype("float32"), (cs[tva + h] > CS_MIN).astype("float32")
    net = Net(Xtr.shape[1])
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    best, bst, pat = 1e9, None, 0
    T = lambda *a: [torch.from_numpy(x) for x in a]
    Xtr_t, Ytr_t, Wtr_t, Xva_t, Yva_t, Wva_t = T(Xtr, Ytr, Wtr, Xva, Yva, Wva)
    for ep in range(60):
        net.train()
        perm = torch.randperm(len(Xtr_t))
        for i in range(0, len(perm), 256):
            b = perm[i:i + 256]
            loss = pinball(net(Xtr_t[b]), Ytr_t[b], Wtr_t[b])
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = float(pinball(net(Xva_t), Yva_t, Wva_t))
        if vl < best - 1e-5:
            best, bst, pat = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            pat += 1
            if pat >= 6:
                break
    net.load_state_dict(bst); net.eval()
    with torch.no_grad():
        qc = net(torch.from_numpy(Xte)).numpy().clip(0, 1.3)
    res[h] = {"persistence_residual_quantiles": score(b1, true, cst), "advection_residual_quantiles": score(b2, true, cst),
              "quantile_cnn": score(qc, true, cst), "cnn_epochs": ep + 1}
    print(h, {k: ({a: round(b, 3) for a, b in v.items()} if isinstance(v, dict) else v) for k, v in res[h].items()}, flush=True)
Path("results").mkdir(exist_ok=True)
json.dump(res, open("results/probabilistic_2025.json", "w"), indent=1)

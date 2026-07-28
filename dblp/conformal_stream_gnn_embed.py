import os, argparse, pickle
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from dgl.nn import GraphConv


def rbf_mmd2(X, Y, gamma):
    Kxx = torch.exp(-gamma * torch.cdist(X, X) ** 2); Kyy = torch.exp(-gamma * torch.cdist(Y, Y) ** 2)
    Kxy = torch.exp(-gamma * torch.cdist(X, Y) ** 2)
    m, n = X.shape[0], Y.shape[0]
    sxx = (Kxx.sum() - Kxx.diag().sum()) / (m * (m - 1)); syy = (Kyy.sum() - Kyy.diag().sum()) / (n * (n - 1))
    return float(sxx + syy - 2 * Kxy.mean())


def bootstrap_threshold(cal_pv, horizon, rng, target_alpha=0.10, n_boot=1000, block=2):
    n = len(cal_pv); maxes = []
    for _ in range(n_boot):
        idx = []
        while len(idx) < horizon:
            start = rng.integers(0, n); idx += [(start + j) % n for j in range(block)]
        idx = idx[:horizon]; boot_pv = cal_pv[idx]
        s = 0.0; m = 0.0
        for p in boot_pv:
            s = max(0.0, s + np.log(max(-np.log(max(p, 1e-12)), 1e-12))); m = max(m, s)
        maxes.append(m)
    return float(np.quantile(maxes, 1 - target_alpha))


def load_graphs(kind, path):
    if kind == "elliptic":
        return pickle.load(open(f"{path}/elliptic_graphs.pkl", "rb"))
    else:
        T = int(pickle.load(open(f"{path}/statistics", "rb"))[0])
        return [pickle.load(open(f"{path}/graph_{i}_by_edges", "rb")) for i in range(T)]


class GCN(nn.Module):
    def __init__(self, in_dim, hid, n_cls, dropout=0.5):
        super().__init__()
        self.c1 = GraphConv(in_dim, hid, allow_zero_in_degree=True)
        self.c2 = GraphConv(hid, n_cls, allow_zero_in_degree=True)
        self.dp = nn.Dropout(dropout)

    def hidden(self, g, x):
        return F.relu(self.c1(g, x))

    def forward(self, g, x):
        return self.c2(g, self.dp(self.hidden(g, x)))


def train_frozen_gcn(graphs, incontrol, hid, epochs, lr, wd, dev, seed=0):
    """Trains on steps 0..incontrol-1 only, using each graph's own labels (y>=0 mask),
    matching the frozen-encoder-trained-on-in-control-data protocol used throughout."""
    torch.manual_seed(seed)
    in_dim = graphs[0].ndata["x"].shape[1]
    n_cls = int(max(int(g.ndata["y"].max()) for g in graphs[:incontrol] if (g.ndata["y"] >= 0).any()) + 1)
    m = GCN(in_dim, hid, n_cls).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    P = [(g.to(dev), g.ndata["x"].float().to(dev), g.ndata["y"].to(dev)) for g in graphs[:incontrol]]
    for ep in range(epochs):
        m.train(); opt.zero_grad()
        losses = []
        for g, x, y in P:
            msk = y >= 0
            if msk.sum() == 0: continue
            losses.append(F.cross_entropy(m(g, x)[msk], y[msk]))
        if losses:
            torch.stack(losses).mean().backward(); opt.step()
    m.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["dblp", "elliptic"])
    ap.add_argument("--path", required=True)
    ap.add_argument("--incontrol", type=int, default=8)
    ap.add_argument("--ref_frac", type=float, default=0.67)
    ap.add_argument("--sub", type=int, default=400)
    ap.add_argument("--hid", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0); rng = np.random.default_rng(0)

    graphs = load_graphs(a.kind, a.path); T = len(graphs)
    model = train_frozen_gcn(graphs, a.incontrol, a.hid, a.epochs, a.lr, a.wd, dev)

    with torch.no_grad():
        feats = [model.hidden(g.to(dev), g.ndata["x"].float().to(dev)).cpu() for g in graphs]

    allX = torch.cat(feats[:a.incontrol], 0); mu = allX.mean(0); sd = allX.std(0) + 1e-6
    feats = [(f - mu) / sd for f in feats]

    def samp(X):
        idx = rng.choice(X.shape[0], min(a.sub, X.shape[0]), replace=False)
        return X[idx]

    n_ref = max(1, int(round(a.incontrol * a.ref_frac))); n_cal = a.incontrol - n_ref
    if n_cal < 1: raise ValueError("incontrol too small for a disjoint ref/calibration split")
    ref_steps = list(range(n_ref)); cal_steps = list(range(n_ref, a.incontrol))
    ref = torch.cat([samp(feats[t]) for t in ref_steps], 0)
    if ref.shape[0] > a.sub * 2:
        ref = ref[rng.choice(ref.shape[0], a.sub * 2, replace=False)]
    with torch.no_grad():
        d = torch.pdist(ref); med = float(d.median()); gamma = 1.0 / (2 * med ** 2 + 1e-9)

    scores = np.array([rbf_mmd2(samp(feats[t]), ref, gamma) for t in range(T)])
    cal = scores[cal_steps]
    pv = np.array([(1 + np.sum(cal >= scores[t])) / (len(cal) + 1) for t in range(T)])
    pv_cal_loo = np.array([(1 + np.sum(np.delete(cal, i) >= cal[i])) / len(cal) for i in range(len(cal))])

    monitor_start = a.incontrol
    cusum = np.full(T, np.nan); s = 0.0
    for t in range(monitor_start, T):
        s = max(0.0, s + np.log(max(-np.log(pv[t]), 1e-12))); cusum[t] = s
    thr_boot = bootstrap_threshold(pv_cal_loo, T - monitor_start, rng, target_alpha=0.10)

    os.makedirs(a.out, exist_ok=True)
    np.savez(f"{a.out}/conformal_{a.kind}_gnnembed.npz", scores=scores, pv=pv, cusum=cusum,
             incontrol=a.incontrol, monitor_start=monitor_start, ref_steps=np.array(ref_steps),
             cal_steps=np.array(cal_steps), thr_illustrative=3.0, thr_bootstrap=thr_boot)
    print(f"=== {a.kind} (GNN-embed)  T={T}  ref={ref_steps}  cal={cal_steps}  monitor_start={monitor_start} ===")
    print(f"bootstrap-calibrated threshold: {thr_boot:.2f}")
    print("t  : MMD    p     CUSUM (nan = before monitoring starts)")
    for t in range(T):
        print(f"{t:>2} : {scores[t]:.3f}  {pv[t]:.3f}  {cusum[t]:.2f}" if t >= monitor_start else
              f"{t:>2} : {scores[t]:.3f}  {pv[t]:.3f}  --")
    print(f"saved {a.out}/conformal_{a.kind}_gnnembed.npz")


if __name__ == "__main__":
    main()

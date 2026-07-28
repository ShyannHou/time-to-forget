import json, os
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, dgl
from dgl.nn import GraphConv
from scipy.spatial.distance import cdist
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
os.makedirs("figures/out", exist_ok=True)

PAPER = {2: ([0.10, 0.08], [0.18, 0.10], 0.05), 3: ([0.70, 0.40, 0.10], [0.50, 0.30, 0.20], 0.03),
         5: ([0.90, 0.70, 0.48, 0.29, 0.10], [0.80, 0.60, 0.38, 0.19, 0.10], 0.01)}
REORDER = {2: [0.05, 0.25], 3: [0.15, 0.45, 0.85], 5: [0.15, 0.35, 0.55, 0.75, 0.95]}
N = 100; CH = 10; POST = 20
K_TO_PLOT = 3  # explicit choice, stated in the caption (paper's original Figure 2 never said)
SEED = 0


def deg(g):
    d_ = g.in_degrees().float(); dm = d_ / d_.max().clamp(min=1)
    return torch.stack([torch.ones_like(d_), dm, dm ** 2, torch.log1p(d_)], 1)


def sbm(base, intra, q, K, rng):
    P = np.full((K, K), q, float)
    for k in range(K): P[k, k] = intra[k]
    pr = P[base[:, None], base[None, :]]; iu = np.triu_indices(N, 1)
    e = rng.random(len(iu[0])) < pr[iu]
    s = np.concatenate([iu[0][e], iu[1][e]]); d_ = np.concatenate([iu[1][e], iu[0][e]])
    g0 = dgl.graph((torch.tensor(s), torch.tensor(d_)), num_nodes=N); x = deg(g0)
    g = dgl.add_self_loop(g0); g.ndata["x"] = x; return g


class GCN(nn.Module):
    def __init__(s, K):
        super().__init__()
        s.c1 = GraphConv(4, 64, allow_zero_in_degree=True); s.c2 = GraphConv(64, K, allow_zero_in_degree=True)
        s.dp = nn.Dropout(0.5)

    def forward(s, g, x): return s.c2(g, s.dp(F.relu(s.c1(g, x))))


def gnn_embed(model, g):
    with torch.no_grad():
        h = F.relu(model.c1(g, g.ndata["x"]))
        return h.mean(0).numpy()


def fit_pointwise_ref(refX):
    mu = refX.mean(0); sd = refX.std(0) + 1e-9; Rz = (refX - mu) / sd
    d = cdist(Rz, Rz); med = np.median(d) + 1e-9; gamma = 1 / (2 * med ** 2)
    Kk = np.exp(-gamma * d ** 2); m = len(Rz)
    krr_term = float((Kk.sum() - np.trace(Kk)) / (m * (m - 1)))
    return {"mu": mu, "sd": sd, "gamma": gamma, "krr_term": krr_term, "Rz": Rz, "m": m}


def pointwise_score(z, ref_stats):
    zz = (z - ref_stats["mu"]) / ref_stats["sd"]
    cross = np.exp(-ref_stats["gamma"] * np.sum((ref_stats["Rz"] - zz) ** 2, axis=1))
    return ref_stats["krr_term"] + 1.0 - (2.0 / ref_stats["m"]) * cross.sum()


K = K_TO_PLOT
p0, p1, q = PAPER[K]; p1r = REORDER[K]
base = np.concatenate([np.full(len(ix), k) for k, ix in enumerate(np.array_split(np.arange(N), K))])

torch.manual_seed(K); rngT = np.random.default_rng(600 + K)
mb_rng = np.random.default_rng(9700 + K)
idxN = np.random.default_rng(12345).permutation(N)
trm = torch.zeros(N, dtype=torch.bool); trm[idxN[:30]] = True
Atr = [sbm(base, p0, q, K, rngT) for _ in range(70)]; yA = torch.tensor(base)
model = GCN(K); opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)
for _ in range(100):
    model.train(); opt.zero_grad(); i = mb_rng.integers(0, 70, 10)
    torch.stack([F.cross_entropy(model(Atr[j], Atr[j].ndata["x"])[trm], yA[trm]) for j in i]).mean().backward()
    opt.step()
model.eval()

rng0 = np.random.default_rng(700 + K)
REF = [sbm(base, p0, q, K, rng0) for _ in range(70)]
with torch.no_grad():
    ref_stats = fit_pointwise_ref(np.array([gnn_embed(model, g) for g in REF]))

# calibration null (same protocol as sbm_pointwise_arl_gnn_embed.py)
r_cal = np.random.default_rng(710 + K)
cal_graphs = [sbm(base, p0, q, K, r_cal) for _ in range(64)]
cal = [pointwise_score(gnn_embed(model, g), ref_stats) for g in cal_graphs]

# the h* value already selected for this K, reorder, in the ARL calibration run
H_STAR = {2: 3.3, 3: 2.2, 5: 2.0}[K]

# one representative change stream: CH in-control + POST post-change (reorder)
rng = np.random.default_rng(3000 + K * 100 + SEED)
stream_graphs = [sbm(base, p0, q, K, rng) for _ in range(CH)] + [sbm(base, p1r, q, K, rng) for _ in range(POST)]
scores = [pointwise_score(gnn_embed(model, g), ref_stats) for g in stream_graphs]

cal_arr = np.array([max(c, 0) for c in cal]); nc = len(cal_arr)
pvals = []; cusum = []; S = 0.0
for sc in scores:
    p = (1 + int((cal_arr >= max(sc, 0)).sum())) / (nc + 1)
    S = max(0.0, S + np.log(max(-np.log(p), 1e-12)))
    pvals.append(p); cusum.append(S)
pvals = np.array(pvals); cusum = np.array(cusum)

fired = next((t for t, c in enumerate(cusum) if c > H_STAR and t >= CH), None)
delay = (fired - CH) if fired is not None else None

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "axes.linewidth": 0.8, "xtick.direction": "in",
    "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
})
NAVY = "#1a3a5c"; MAROON = "#7a1f1f"

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.4, 5.2), sharex=True)
t = np.arange(len(scores))
ax1.plot(t, pvals, marker="o", markersize=3.5, linewidth=1.1, color=NAVY)
ax1.axvline(CH - 0.5, ls="--", c="0.4", lw=0.9)
ax1.set_ylabel("conformal $p$-value", fontsize=11)
ax1.grid(alpha=0.25, linewidth=0.5)

ax2.plot(t, cusum, marker="o", markersize=3.5, linewidth=1.1, color=MAROON)
ax2.axhline(H_STAR, ls="--", c="0.15", lw=0.9)
ax2.axvline(CH - 0.5, ls="--", c="0.4", lw=0.9)
ax2.text(CH - 0.5, ax2.get_ylim()[1] * 0.92, " change", fontsize=8.5, color="0.3")
ax2.text(0.3, H_STAR + ax2.get_ylim()[1] * 0.02, f"$h^*={H_STAR}$", fontsize=8.5, color="0.15")
if fired is not None:
    ax2.axvline(fired, ls=":", c=MAROON, lw=1.2)
    ax2.text(fired + 0.3, ax2.get_ylim()[1] * 0.55, f"alarm\n(delay={delay})", fontsize=8.5, color=MAROON)
ax2.set_ylabel("CUSUM $C_t$", fontsize=11); ax2.set_xlabel("stream time step", fontsize=11)
ax2.grid(alpha=0.25, linewidth=0.5)
for ax in (ax1, ax2):
    ax.tick_params(labelsize=9.5)
fig.tight_layout()
fig.savefig("figures/out/figure2_gnnembed_pointwise.png", dpi=200, bbox_inches="tight")
np.savez("results/sbm/figure2_gnnembed_pointwise.npz", pvals=pvals, cusum=cusum, scores=np.array(scores),
         h_star=H_STAR, change_time=CH, fired=fired if fired is not None else -1, K=K, seed=SEED)
print(f"K={K} seed={SEED}: fired={fired} delay={delay} h*={H_STAR}")
print("saved figures/out/figure2_gnnembed_pointwise.png and results/sbm/figure2_gnnembed_pointwise.npz")

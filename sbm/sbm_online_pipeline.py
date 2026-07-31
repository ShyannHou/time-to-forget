import json, argparse
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, dgl
from dgl.nn import GraphConv
from scipy.spatial.distance import cdist

N = 100
PAPER = {2: ([0.10, 0.08], [0.18, 0.10], 0.05), 3: ([0.70, 0.40, 0.10], [0.50, 0.30, 0.20], 0.03),
         5: ([0.90, 0.70, 0.48, 0.29, 0.10], [0.80, 0.60, 0.38, 0.19, 0.10], 0.01)}
REORDER = {2: [0.05, 0.25], 3: [0.15, 0.45, 0.85], 5: [0.15, 0.35, 0.55, 0.75, 0.95]}

ENCODER_TRAIN_SIZE = 70
INITIAL_REF_SIZE = 70
CONNECTIVITY_INITIAL_CAL_SIZE = 64
RENEW_REF_SIZE = 70
RENEW_CAL_SIZE = 20
BURN_IN = RENEW_REF_SIZE + RENEW_CAL_SIZE   # 90
TARGET_ARL = 100
MC_THRESH = 80   # threshold-(re)selection Monte Carlo runs per calibration; raised from
                 # the original 30 after full-scale runs showed MC=30 gives an unreliable
                 # threshold (both initially and after renewal), matching the fix already
                 # validated in sbm_pointwise_arl_gnn_embed_highmc.py


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
    dev = next(model.parameters()).device
    with torch.no_grad():
        h = F.relu(model.c1(g.to(dev), g.ndata["x"].to(dev)))
        return h.mean(0).cpu().numpy()


def fit_pointwise_ref(refX):
    mu = refX.mean(0); sd = refX.std(0) + 1e-9; Rz = (refX - mu) / sd
    m = len(Rz); iu = np.triu_indices(m, 1)
    ref_pair_sq = cdist(Rz, Rz)[iu] ** 2
    return {"mu": mu, "sd": sd, "Rz": Rz, "m": m, "ref_pair_sq": ref_pair_sq}


def pointwise_score(z, ref_stats):
    zz = (z - ref_stats["mu"]) / ref_stats["sd"]
    Rz = ref_stats["Rz"]; m = ref_stats["m"]; ref_pair_sq = ref_stats["ref_pair_sq"]
    d_to_z = np.sqrt(np.sum((Rz - zz) ** 2, axis=1))
    all_d = np.concatenate([np.sqrt(ref_pair_sq), d_to_z])
    med = np.median(all_d) + 1e-9; gamma = 1 / (2 * med ** 2)
    cross = np.exp(-gamma * d_to_z ** 2)
    krr_term = 2.0 * np.exp(-gamma * ref_pair_sq).sum() / (m * (m - 1))
    return krr_term + 1.0 - (2.0 / m) * cross.sum()


def cusum_step(prev_S, sc, cal):
    cal = np.array([max(c, 0) for c in cal]); nc = len(cal)
    p = (1 + int((cal >= max(sc, 0)).sum())) / (nc + 1)
    return max(0.0, prev_S + np.log(max(-np.log(p), 1e-12)))


def calibrate_threshold(model, ref_stats, cal, base, in_control_intra, q, K, rng,
                         target_arl=TARGET_ARL, mc=MC_THRESH, horizon=600):
    hs = np.arange(0.2, 10, 0.1); arl = {h: [] for h in hs}
    for r in range(mc):
        srng = np.random.default_rng(rng.integers(0, 2 ** 31))
        gs = [sbm(base, in_control_intra, q, K, srng) for _ in range(horizon)]
        scores = [pointwise_score(gnn_embed(model, g), ref_stats) for g in gs]
        for h in hs:
            S = 0.0; fired = None
            for t, sc in enumerate(scores):
                S = cusum_step(S, sc, cal)
                if S > h: fired = t; break
            arl[h].append(fired if fired is not None else horizon)
    arl_mean = {h: np.mean(arl[h]) for h in hs}
    return float(min(hs, key=lambda h: abs(arl_mean[h] - target_arl)))


def train_gcn(graphs, labels, epochs, mb, mb_rng, dev, init=None, weights=None):
    K = int(labels.max().item()) + 1
    m = GCN(K).to(dev)
    if init is not None: m.load_state_dict(init)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2, weight_decay=5e-4)
    idx = np.arange(len(graphs)); w = np.ones(len(graphs)) if weights is None else np.asarray(weights)
    mask = torch.ones(N, dtype=torch.bool)   # full-node supervision; fresh graph draws each eval, no leakage
    for _ in range(epochs):
        m.train(); mb_rng.shuffle(idx); sel = idx[:min(mb, len(idx))]; opt.zero_grad()
        losses = [w[i] * F.cross_entropy(m(graphs[i].to(dev), graphs[i].ndata["x"].to(dev))[mask],
                                          labels.to(dev)[mask]) for i in sel]
        torch.stack(losses).mean().backward(); opt.step()
    m.eval()
    return m


def eval_acc(model, g, labels, dev):
    with torch.no_grad():
        pr = model(g.to(dev), g.ndata["x"].to(dev)).argmax(1)
        return float((pr == labels.to(dev)).float().mean())


def run_seed(K, seed, policy, len_A=100, len_B=150, len_C=150, dev=None):
    dev = dev or torch.device("cpu")
    p0, p1r, q = PAPER[K][0], REORDER[K], PAPER[K][2]
    base = np.concatenate([np.full(len(ix), k) for k, ix in enumerate(np.array_split(np.arange(N), K))])
    labels = torch.tensor(base)

    torch.manual_seed(seed)
    setup_rng = np.random.default_rng(4000 + K * 100 + seed)
    mb_rng = np.random.default_rng(9000 + K * 100 + seed)
    stream_rng = np.random.default_rng(5000 + K * 100 + seed)

    Atr = [sbm(base, p0, q, K, setup_rng) for _ in range(ENCODER_TRAIN_SIZE)]
    model = train_gcn(Atr, labels, epochs=100, mb=10, mb_rng=mb_rng, dev=dev)
    REF = [sbm(base, p0, q, K, setup_rng) for _ in range(INITIAL_REF_SIZE)]
    ref_stats = fit_pointwise_ref(np.array([gnn_embed(model, g) for g in REF]))
    CAL = [sbm(base, p0, q, K, setup_rng) for _ in range(CONNECTIVITY_INITIAL_CAL_SIZE)]
    cal = [pointwise_score(gnn_embed(model, g), ref_stats) for g in CAL]
    h_star = calibrate_threshold(model, ref_stats, cal, base, p0, q, K, setup_rng)

    stream_specs = ([p0] * len_A + [p1r] * len_B + [p0] * len_C)
    stream_graphs = [sbm(base, intra, q, K, stream_rng) for intra in stream_specs]
    stream_incontrol = [p0] * len_A + [p1r] * len_B + [p0] * len_C   # the CURRENT regime's own distribution
    true_changes = [len_A, len_A + len_B]

    S = 0.0; accs = []
    burn_in_until = None; burn_in_start = None; pre_alarm_graphs = []
    cycles = []; next_change_ix = 0; retrain_ix = 0

    for t, g in enumerate(stream_graphs):
        if burn_in_until is not None and t < burn_in_until:
            accs.append(eval_acc(model, g, labels, dev))
            continue

        if burn_in_until is not None and t == burn_in_until:
            post_graphs = stream_graphs[burn_in_start:burn_in_until]
            pre_graphs = cycles[-1]["_pre_graphs"]
            if policy == "no-forget":
                train_graphs = pre_graphs + post_graphs; weights = None; init = None
            elif policy == "detected-soft":
                train_graphs = pre_graphs + post_graphs
                weights = [0.3] * len(pre_graphs) + [1.0] * len(post_graphs)
                init = {k: v.clone() for k, v in model.state_dict().items()}
            else:   # oracle-hard, detected-hard
                train_graphs = post_graphs if post_graphs else pre_graphs[-BURN_IN:]
                weights = None; init = None

            if len(post_graphs) >= RENEW_REF_SIZE + RENEW_CAL_SIZE:
                new_ref_graphs = post_graphs[:RENEW_REF_SIZE]
                new_cal_graphs = post_graphs[RENEW_REF_SIZE:RENEW_REF_SIZE + RENEW_CAL_SIZE]
            else:
                half = max(1, len(post_graphs) // 2)
                new_ref_graphs, new_cal_graphs = post_graphs[:half], post_graphs[half:]

            model = train_gcn(train_graphs, labels, epochs=100, mb=10,
                               mb_rng=np.random.default_rng(seed * 1000 + retrain_ix + 1), dev=dev,
                               init=init, weights=weights)
            if new_ref_graphs and new_cal_graphs:
                ref_stats = fit_pointwise_ref(np.array([gnn_embed(model, gg) for gg in new_ref_graphs]))
                cal = [pointwise_score(gnn_embed(model, gg), ref_stats) for gg in new_cal_graphs]
                h_star = calibrate_threshold(model, ref_stats, cal, base, stream_incontrol[t], q, K,
                                              np.random.default_rng(seed * 2000 + retrain_ix + 1))
            S = 0.0; burn_in_until = None; burn_in_start = None; retrain_ix += 1

        acc_t = eval_acc(model, g, labels, dev)
        accs.append(acc_t)
        pre_alarm_graphs.append(g)

        if policy == "oracle-hard":
            fire = next_change_ix < len(true_changes) and t == true_changes[next_change_ix]
        else:
            z = gnn_embed(model, g)
            sc = pointwise_score(z, ref_stats)
            S = cusum_step(S, sc, cal)
            fire = S > h_star

        if fire and burn_in_until is None:
            true_t = true_changes[next_change_ix] if next_change_ix < len(true_changes) else None
            crossed = (next_change_ix + 1 < len(true_changes) and t + 1 + BURN_IN > true_changes[next_change_ix + 1])
            cycles.append({"true_change_time": true_t, "alarm_time": t,
                            "detection_delay": (t - true_t) if true_t is not None else None,
                            # an alarm with no true change left to correspond to (all true
                            # changes already triggered a prior cycle) is also a false alarm,
                            # not the default False that "true_t is None" would otherwise imply
                            "false_alarm": (true_t is None or t < true_t),
                            "burn_in_crossed_regime": crossed, "renewal_start_index": t,
                            "_pre_graphs": pre_alarm_graphs})
            burn_in_start = t + 1; burn_in_until = t + 1 + BURN_IN
            pre_alarm_graphs = []; next_change_ix += 1

    # post-process: slice the already-computed accuracy trace per cycle (no re-evaluation needed)
    for i, cyc in enumerate(cycles):
        cyc.pop("_pre_graphs")
        renew_end = min(cyc["renewal_start_index"] + 1 + BURN_IN, len(stream_graphs))
        window_end = cycles[i + 1]["alarm_time"] if i + 1 < len(cycles) else len(stream_graphs)
        post_accs = accs[renew_end:window_end]
        steady = float(np.mean(post_accs[-10:])) if len(post_accs) >= 10 else (float(np.mean(post_accs)) if post_accs else None)
        recovery = None
        if steady is not None:
            for j, acc in enumerate(post_accs):
                if acc >= 0.9 * steady: recovery = j; break
        cyc["mean_post_change_accuracy"] = float(np.mean(post_accs)) if post_accs else None
        cyc["steady_state_accuracy"] = steady
        cyc["recovery_time"] = recovery

    return {"policy": policy, "config": {"K": K, "seed": seed, "len_A": len_A, "len_B": len_B, "len_C": len_C},
            "accuracy_by_time": accs, "cycles": cycles, "n_alarms": len(cycles)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=3, choices=[2, 3, 5])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--len_A", type=int, default=100)
    ap.add_argument("--len_B", type=int, default=150)
    ap.add_argument("--len_C", type=int, default=150)   # must exceed ~BURN_IN(90)+detection_delay for the
                                                          # second renewal cycle to actually complete inline
    ap.add_argument("--out", default="results/sbm/sbm_online_pipeline_results.json")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policies = ["oracle-hard", "detected-hard", "detected-soft", "no-forget"]
    all_results = {p: [] for p in policies}
    for seed in range(a.seeds):
        for p in policies:
            r = run_seed(a.K, seed, p, a.len_A, a.len_B, a.len_C, dev)
            all_results[p].append(r)
        print(f"seed {seed} done", flush=True)

    summary = {}
    for p, recs in all_results.items():
        first_delays = [c["cycles"][0]["detection_delay"] for c in recs if c["cycles"] and c["cycles"][0]["detection_delay"] is not None]
        first_accs = [c["cycles"][0]["mean_post_change_accuracy"] for c in recs if c["cycles"] and c["cycles"][0].get("mean_post_change_accuracy") is not None]
        n_alarms = [c["n_alarms"] for c in recs]
        all_flags = [cyc["false_alarm"] for c in recs for cyc in c["cycles"]]
        summary[p] = {"mean_first_detection_delay": float(np.mean(first_delays)) if first_delays else None,
                       "mean_first_cycle_post_change_accuracy": float(np.mean(first_accs)) if first_accs else None,
                       "mean_n_alarms": float(np.mean(n_alarms)),
                       "false_alarm_rate_over_all_cycles": float(np.mean(all_flags)) if all_flags else None}
        print(f"{p:<16} {summary[p]}")

    json.dump({"per_seed": all_results, "summary": summary}, open(a.out, "w"), indent=2)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()

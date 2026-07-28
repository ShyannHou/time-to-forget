import os, sys, argparse, pickle, json
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import dgl
from dgl.nn import GraphConv

def load(D, name):
    with open(os.path.join(D, name), "rb") as f:
        return pickle.load(f)

def masks_of(mask_tuple):
    tr, va, te = mask_tuple
    b = lambda t: t.bool() if t.dtype != torch.bool else t
    return b(tr), b(va), b(te)

class GCN(nn.Module):
    def __init__(self, in_dim, hid, n_cls, dropout=0.5):
        super().__init__()
        self.c1 = GraphConv(in_dim, hid, allow_zero_in_degree=True)
        self.c2 = GraphConv(hid, n_cls, allow_zero_in_degree=True)
        self.dp = nn.Dropout(dropout)
    def forward(self, g, x):
        h = F.relu(self.c1(g, x)); h = self.dp(h)
        return self.c2(g, h)

def train_model(specs, val_specs, in_dim, n_cls, args, dev, init_state=None):
    """specs / val_specs: list of (g, x, y, mask). Joint multi-graph training."""
    model = GCN(in_dim, args.hidden, n_cls, args.dropout).to(dev)
    if init_state is not None:
        model.load_state_dict(init_state)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    best_va, best_state = -1.0, None
    for ep in range(args.epochs):
        model.train(); opt.zero_grad()
        losses = []
        for g, x, y, m in specs:
            out = model(g, x)
            losses.append(F.cross_entropy(out[m], y[m]))
        loss = torch.stack(losses).mean()
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            corr = tot = 0
            for g, x, y, m in val_specs:
                p = model(g, x)[m].argmax(1); corr += (p == y[m]).sum().item(); tot += m.sum().item()
            va = corr / max(tot, 1)
        if va > best_va:
            best_va = va; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best_va

@torch.no_grad()
def acc_on(model, specs):
    model.eval(); corr = tot = 0
    for g, x, y, m in specs:
        p = model(g, x)[m].argmax(1); corr += (p == y[m]).sum().item(); tot += m.sum().item()
    return corr / max(tot, 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dblp")
    ap.add_argument("--pairs", default="0:9;0:8;1:9;8:9;0,1,2:7,8,9")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results/dblp")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.pairs = "0:9"; args.seeds = 1; args.epochs = 30
    D, dev = args.data, torch.device(args.device)
    n_cls = int(load(D, "statistics")[1])
    pairs = [(list(map(int, a.split(","))), list(map(int, b.split(",")))) for a, b in
             (p.split(":") for p in args.pairs.split(";"))]

    # cache prepped graphs (self-loop, on device); reload masks per seed
    subs = {}
    def prep(i):
        if i not in subs:
            g = dgl.add_self_loop(dgl.remove_self_loop(load(D, f"sub_graph_{i}_by_edges"))).to(dev)
            subs[i] = (g, g.ndata["x"].float(), g.ndata["y"].long())
        return subs[i]
    in_dim = prep(0)[1].shape[1]

    def specs_for(idxs, seed, which):  # which: 0=train,1=val,2=test
        out = []
        M = load(D, f"mask_seed_{seed}")
        for i in idxs:
            g, x, y = prep(i)
            out.append((g, x, y, masks_of(M[i])[which].to(dev)))
        return out

    ARMS = ["stale_A", "cumulative", "finetune", "forget"]
    results = {}  # (pairstr) -> arm -> list of (accB, accA) over seeds
    for A, B in pairs:
        ps = f"{'+'.join(map(str,A))}->{'+'.join(map(str,B))}"
        results[ps] = {a: [] for a in ARMS}
        for s in range(args.seeds):
            torch.manual_seed(s); np.random.seed(s)
            A_tr, A_va, A_te = specs_for(A, s, 0), specs_for(A, s, 1), specs_for(A, s, 2)
            B_tr, B_va, B_te = specs_for(B, s, 0), specs_for(B, s, 1), specs_for(B, s, 2)

            m_stale, _ = train_model(A_tr, A_va, in_dim, n_cls, args, dev)
            m_cumul, _ = train_model(A_tr + B_tr, A_va + B_va, in_dim, n_cls, args, dev)
            m_ft, _    = train_model(B_tr, B_va, in_dim, n_cls, args, dev,
                                     init_state={k: v.clone() for k, v in m_stale.state_dict().items()})
            m_forget, _= train_model(B_tr, B_va, in_dim, n_cls, args, dev)

            for name, mdl in [("stale_A", m_stale), ("cumulative", m_cumul),
                              ("finetune", m_ft), ("forget", m_forget)]:
                results[ps][name].append((acc_on(mdl, B_te), acc_on(mdl, A_te)))
        # report this pair
        print(f"\n=== splice {ps}   (seeds={args.seeds}, epochs={args.epochs}) ===")
        print(f"{'arm':<12} {'acc_B(new)':>12} {'acc_A(old)':>12}")
        mB = {}
        for a in ARMS:
            bs = np.array([r[0] for r in results[ps][a]]); As = np.array([r[1] for r in results[ps][a]])
            mB[a] = bs.mean()
            print(f"{a:<12} {bs.mean():>7.3f}±{bs.std():.3f} {As.mean():>7.3f}±{As.std():.3f}")
        print(f"  >> forget gain vs cumulative (on B): {mB['forget']-mB['cumulative']:+.3f}")
        print(f"  >> forget gain vs stale_A    (on B): {mB['forget']-mB['stale_A']:+.3f}")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "forget_results.pkl"), "wb") as f:
        pickle.dump(results, f)

    json_out = {"config": {"seeds": args.seeds, "epochs": args.epochs, "hidden": args.hidden,
                            "lr": args.lr, "wd": args.wd, "pairs": args.pairs}, "per_pair": {}}
    for ps in results:
        entry = {"per_seed": {}, "mean": {}, "std": {}}
        for a in ARMS:
            bs = [float(r[0]) for r in results[ps][a]]
            entry["per_seed"][a] = bs
            entry["mean"][a] = float(np.mean(bs))
            entry["std"][a] = float(np.std(bs))
        entry["forget_minus_cumulative"] = entry["mean"]["forget"] - entry["mean"]["cumulative"]
        entry["forget_minus_stale"] = entry["mean"]["forget"] - entry["mean"]["stale_A"]
        json_out["per_pair"][ps] = entry
    with open(os.path.join(args.out, "dblp_forget_results.json"), "w") as f:
        json.dump(json_out, f, indent=2)

    # bar chart: acc_B per arm per pair
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        pnames = list(results.keys()); x = np.arange(len(pnames)); w = 0.2
        fig, ax = plt.subplots(figsize=(1.8*len(pnames)+3, 4.5))
        for k, a in enumerate(ARMS):
            means = [np.mean([r[0] for r in results[p][a]]) for p in pnames]
            errs  = [np.std([r[0] for r in results[p][a]]) for p in pnames]
            ax.bar(x + (k-1.5)*w, means, w, yerr=errs, capsize=3, label=a)
        ax.set_xticks(x); ax.set_xticklabels(pnames)
        ax.set_ylabel("accuracy on NEW regime B"); ax.set_ylim(0.5, 1.0)
        ax.set_title(f"Time to Forget: downstream acc on new regime (seeds={args.seeds})")
        ax.legend(ncol=4, fontsize=8); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(args.out, "forget_experiment.png"), dpi=130)
        print(f"\nsaved -> {args.out}/forget_experiment.png + forget_results.pkl")
    except Exception as e:
        print("plot skipped:", e)

if __name__ == "__main__":
    main()

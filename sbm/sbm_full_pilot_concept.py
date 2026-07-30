"""Pilot-graph concept-label construction: addresses the retrospective /
transductive labeling issue in sbm_full.py's "concept" change type, where
degree_labels_avg(B, K) averages degree over the SAME B graphs later used for
training/monitoring/test -- so even the first B graph's label depends on B
graphs that, in an online setting, have not arrived yet.

Fix: the label rule is instead computed once from an INDEPENDENT pilot set of
graphs (drawn from the same regime-B distribution, via a disjoint RNG stream
that never enters training/monitoring/test), then applied as a fixed rule to
the actual B graphs before any of them are used.

Status: written but NOT executed. This does NOT replace sbm_full.py, which
remains the source of the currently-reported Table 2 concept-change numbers
(computed with the retrospective construction) -- running this script and
adopting its labels would change those numbers and requires a full rerun plus
a paper update, which is out of scope given time constraints. Until that
rerun happens, the retrospective-labeling caveat should be disclosed in the
paper/README rather than silently fixed here. Reuses gen/degree_labels_avg/
GCN/train/evalacc from sbm_full.py unchanged.
"""
import argparse, json
import numpy as np, torch
from sbm_full import N, NG, PAPER, CBASE, gen, degree_labels_avg, GCN, train, evalacc


def pilot_concept_labels(base, cp0, q, K, rng, n_pilot=100):
    pilot = gen(base, cp0, q, K, n_pilot, rng)
    return degree_labels_avg(pilot, K)


def run(dev, seeds=5, ep=100):
    idx = np.random.default_rng(12345).permutation(N)
    trm = torch.zeros(N, dtype=torch.bool); trm[idx[:30]] = True
    vam = torch.zeros(N, dtype=torch.bool); vam[idx[30:45]] = True
    tem = torch.zeros(N, dtype=torch.bool); tem[idx[60:]] = True
    R = {}
    for K in (2, 3, 5):
        _, _, q = PAPER[K]
        base = np.concatenate([np.full(len(ix), k) for k, ix in enumerate(np.array_split(np.arange(N), K))])
        cp0 = CBASE[K]
        Sc = []; Cu = []; Fo = []
        for s in range(seeds):
            torch.manual_seed(s)
            rng = np.random.default_rng(700 + K * 10 + s)
            pilot_rng = np.random.default_rng(50000 + K * 10 + s)   # disjoint from rng: never shared with A/B draws
            A = gen(base, cp0, q, K, NG, rng); yA = torch.tensor(base)
            lab = pilot_concept_labels(base, cp0, q, K, pilot_rng)   # label rule fixed BEFORE any B graph is drawn
            B = gen(base, cp0, q, K, NG, rng); yB = torch.tensor(lab)
            yAl = [yA] * NG; yBl = [yB] * NG

            base_state = {k: v.clone() for k, v in GCN(K).to(dev).state_dict().items()}
            def mbr(_s=s, _K=K): return np.random.default_rng(9000 + _K * 10 + _s)
            st = train(A, yAl, trm, A, yAl, vam, K, dev, ep, mb_rng=mbr(), init=base_state)
            Sc.append(evalacc(st, B, yBl, tem, dev))
            cu = train(A + B, yAl + yBl, trm, B, yBl, vam, K, dev, ep, mb_rng=mbr(), init=base_state)
            Cu.append(evalacc(cu, B, yBl, tem, dev))
            fo = train(B, yBl, trm, B, yBl, vam, K, dev, ep, mb_rng=mbr(), init=base_state)
            Fo.append(evalacc(fo, B, yBl, tem, dev))
        R[f"{K}-concept-pilot"] = {
            "stale": float(np.mean(Sc)), "stale_std": float(np.std(Sc)),
            "cumulative": float(np.mean(Cu)), "cumulative_std": float(np.std(Cu)),
            "forget": float(np.mean(Fo)), "forget_std": float(np.std(Fo)),
            "stale_per_seed": [float(x) for x in Sc],
            "cumulative_per_seed": [float(x) for x in Cu],
            "forget_per_seed": [float(x) for x in Fo]}
        print(f"[K={K} concept-pilot] st/cu/fo={np.mean(Sc):.3f}/{np.mean(Cu):.3f}/{np.mean(Fo):.3f}", flush=True)
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--ep", type=int, default=100)
    ap.add_argument("--out", default="results/sbm/sbm_full_pilot_concept_results.json")
    a = ap.parse_args()
    R = run(torch.device(a.device), a.seeds, a.ep)
    json.dump(R, open(a.out, "w"), indent=2)
    print("saved", a.out)


if __name__ == "__main__":
    main()

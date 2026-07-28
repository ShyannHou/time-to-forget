# Time to Forget: Continual Learning on Dynamic Graphs via Regime Change Detection

Code for the anonymous submission *"Time to Forget: Continual Learning on Dynamic Graphs
via Regime Change Detection."*

The method monitors a stream of graph representations with a **conformal p-value +
CUSUM** chart. When the chart alarms, the monitoring reference is renewed and the
continual learner applies a memory policy (no-forget / soft-forget / hard-forget). This
repository contains the code for all reported experiments, plus the saved result files so
every number and figure in the paper can be checked without rerunning anything.

---

## 1. Repository layout

```
sbm/          synthetic stochastic-block-model experiments (no external data needed)
elliptic/     Elliptic (Bitcoin) real-data experiments
dblp/         DBLP negative-control detection
figures/      plotting scripts; all read from results/, none hardcode numbers
results/      saved JSON / NPZ outputs of every script below
```

## 2. Requirements

```
python >= 3.9
torch, dgl            # GCN via dgl.nn.GraphConv
numpy, scipy          # scipy.spatial.distance.cdist for the MMD kernel
matplotlib            # figures only
```

The SBM experiments run on CPU in a few minutes each. The Elliptic experiments were run
on a single GPU.

## 3. Data

**SBM** needs no external data: graphs are generated on the fly from a fixed seed.

**Elliptic** expects `data/elliptic/elliptic_graphs.pkl`, a pickled list of 49 DGL graphs
with `ndata["x"]` (165 features) and `ndata["y"]` (1 = illicit, 0 = licit, -1 = unknown),
built from the public
[Elliptic Data Set](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set).
Override the location with `--data` (or `ELLIPTIC_DIR` for `elliptic_policy.py`).

**DBLP** expects a directory of per-year snapshots (`sub_graph_<i>_by_edges` plus a
`statistics` file giving the number of snapshots), derived from the public
[ArnetMiner/DBLP](https://www.aminer.org/citation) citation data. Pass it with `--path`.

Both real datasets are redistributed by their original providers, not here.

## 4. Reproducing the experiments

### 4.1 Synthetic SBM

```bash
python sbm/sbm_full.py                    --out results/sbm/sbm_full_results.json
python sbm/sbm_pointwise_arl_gnn_embed.py # primary: pointwise score (Eq. 2) on a task-trained GCN's embedding
python sbm/sbm_pointwise_arl.py          # secondary: pointwise score on the hand-crafted 8-d degree summary
python sbm/sbm_arl2.py                    # secondary: windowed (5-snapshot) score, hand-crafted feature
python sbm/sbm_policy.py                  # no-/soft-/hard-forget + warm start
python sbm/sbm_ratio3.py                  # equal-training-size control sweep
```

`sbm_full.py` produces, for K in {2,3,5} and four change types, the unsupervised MMD
score, the frozen-model supervised loss, alarm times, and downstream stale / cumulative /
forget accuracy over 5 seeds.

### 4.2 Elliptic

```bash
python elliptic/elliptic_forget.py --A 1-42 --B 43-49 --out results/elliptic/elliptic_forget_sharp.json
python elliptic/elliptic_forget.py --A 1-34 --B 35-49 --out results/elliptic/elliptic_forget_broad.json
python elliptic/elliptic_ratio.py  --out results/elliptic/elliptic_ratio.json
python elliptic/elliptic_policy.py
# supervised detector, short and event-aligned windows, 5 seeds each
for s in 0 1 2 3 4; do
  python elliptic/elliptic_sup_detect.py --train_end 20 --cal_end 30 --seed $s
  python elliptic/elliptic_sup_detect.py --train_end 30 --cal_end 42 --seed $s
done
# unsupervised feature-MMD detector, raw node features
python dblp/conformal_stream.py --kind elliptic --path data/elliptic --incontrol 30 --out results/elliptic
python dblp/conformal_stream.py --kind elliptic --path data/elliptic --incontrol 42 --out results/elliptic
# unsupervised feature-MMD detector, task-trained GCN embedding
python dblp/conformal_stream_gnn_embed.py --kind elliptic --path data/elliptic --incontrol 10 --out results/elliptic
```

### 4.3 DBLP (negative control)

```bash
python dblp/conformal_stream.py --kind dblp --path <dblp-snapshot-dir> --incontrol 6 --out results/dblp
python dblp/conformal_stream_gnn_embed.py --kind dblp --path <dblp-snapshot-dir> --incontrol 6 --out results/dblp
```

### 4.4 Figures

```bash
python figures/plot_sbm_perclass.py       # per-class SBM detection + downstream panels
python figures/plot_concept_merged.py     # concept change merged across K
python figures/plot_experiments.py        # post-alarm memory policy
python figures/plot_ratio.py              # equal-training-size sweeps (SBM + Elliptic)
python figures/plot_elliptic.py           # Elliptic downstream forgetting
python figures/plot_elliptic_detect.py    # Elliptic/DBLP evidence trajectories
python figures/plot_elliptic_detect2.py   # short vs event-aligned reference window
python figures/plot_elliptic_multiseed.py # 5-seed overlay of the supervised detector
python figures/plot_figure2_gnnembed.py   # single representative run, pointwise GCN-embedding detector
python figures/plot_figure5.py            # Elliptic supervised (1-recall) evidence trajectory
```

Plotting scripts read from `results/` and derive every plotted value and annotation from
those files, so a rerun of an upstream script is reflected in the figures automatically.

## 5. Method details worth knowing when reading the code

- **Conformal p-value** (Eq. 1) and the **CUSUM recursion** `C_t = max(0, C_{t-1} +
  log f(p_t))` with log-surprisal betting `f(p) = -log p` are implemented in
  `cusum_fire` / `run_cusum` / `monitor` across the scripts.
- **Two nonconformity scores** are used: a label-free kernel-MMD score on graph
  representations, and a supervised score (frozen-model cross-entropy on SBM,
  `1 - illicit-recall` on Elliptic) for changes that alter the labeling rule rather than
  the structure.
- **Pointwise vs windowed.** The literal Eq. 2 single-snapshot score is the primary
  detector reported in the paper (`sbm_pointwise_arl_gnn_embed.py`); a 5-snapshot
  sliding-window variant (`sbm_arl2.py`) is lower-variance but makes consecutive scores
  dependent, and is kept as a secondary comparison.
- **Representation.** For SBM's connectivity-change detector, the reported score is a
  kernel-MMD comparison on a task-trained GCN's mean-pooled hidden-layer embedding
  (`sbm_pointwise_arl_gnn_embed.py`), matching the paper's z_t = f_phi(G_t) framing. A
  hand-crafted 8-dimensional degree-distribution summary (`sbm_pointwise_arl.py`,
  `sbm_arl2.py`) is kept as a comparison baseline -- across K the two representations
  trade off (faster at some K, slower at others; see the per-K delay/ARL0 numbers in
  `results/sbm/`), so the embedding was adopted for consistency with the paper's stated
  method rather than for a uniform detection-speed win.
- **Data-split discipline.** Training, reference, calibration and monitoring blocks are
  kept disjoint; standardization statistics are computed only from pre-monitoring data;
  the CUSUM is reset at the true monitoring start rather than accumulated from step 1.
  On SBM the supervised monitor scores on a node mask disjoint from both the classifier's
  train/validation masks and the downstream test mask.
- **Thresholds.** SBM thresholds are Monte-Carlo calibrated to a target in-control
  ARL0 ≈ 100, with threshold selection and ARL0 evaluation on independent simulation
  batches. On real data there is no known in-control generating model, so no threshold is
  claimed to achieve a verified ARL0; the real-data figures show raw evidence
  trajectories and threshold-dependent alarm times are reported as a sensitivity analysis.

## 6. Scope and limitations

- The Elliptic/DBLP unsupervised monitor's primary reported result
  (`dblp/conformal_stream.py`) uses an MMD on the raw node-feature distribution, not a
  trained-GNN embedding. A GCN-embedding variant is also included
  (`dblp/conformal_stream_gnn_embed.py`, `results/{elliptic,dblp}/*_gnnembed.npz`) for
  comparison: it does not change the qualitative behavior (both representations show a
  steadily rising, non-event-specific trend on Elliptic's full-stream reference window),
  indicating the short reference/monitoring-window length is the limiting factor here,
  not the choice of representation.
- Topology-augmented (zigzag-persistence) representations are **not** exercised by any
  script here.
- All results use 5 seeds; means and per-seed values are stored in `results/`, but no
  paired-difference confidence intervals are computed.
- The Elliptic supervised detector with the short (20-step) training window is
  seed-sensitive; the 5-seed spread is saved in
  `results/elliptic/elliptic_sup_detect_t20c30_seed*.npz` and shown by
  `figures/plot_elliptic_multiseed.py`. Across the 5 seeds the detector never fires
  before the true event and always fires within a few steps after it, but the final
  CUSUM magnitude varies roughly 2x seed to seed.
- The pointwise ARL0 calibration does not hit the target of 100 exactly at every K; the
  achieved ARL0 (independent evaluation batch) ranges from about 64 to 115 across
  K in {2,3,5} for the GCN-embedding detector. Both the target and achieved values are
  saved per-K in `results/sbm/sbm_pointwise_arl_gnn_embed_results.json`.

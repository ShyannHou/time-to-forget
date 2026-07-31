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
sbm/               synthetic stochastic-block-model experiments (no external data needed)
elliptic/          Elliptic (Bitcoin) real-data experiments + data builder
dblp/              DBLP detection and downstream forgetting experiments
figures/           plotting scripts; all read from results/, none hardcode numbers
results/           saved JSON / NPZ outputs of every script below
requirements.txt   exact package versions (see ENVIRONMENT.md)
ENVIRONMENT.md     Python/CUDA/package versions, exported from the real run environment
DATA.md            exact Elliptic/DBLP file formats and construction rules
```

## 2. Requirements

Exact versions that produced every saved result in `results/` are in
[`ENVIRONMENT.md`](ENVIRONMENT.md) / [`requirements.txt`](requirements.txt) (`torch`,
`dgl`, `numpy`, `scipy`, `matplotlib`, `scikit-learn`, Python 3.9, CUDA 12.1).

The SBM experiments run on CPU in a few minutes each. The Elliptic experiments were run
on a single GPU.

## 3. Data

**SBM** needs no external data: graphs are generated on the fly from a fixed seed.

**Elliptic** and **DBLP** are redistributed by their original providers, not here. See
[`DATA.md`](DATA.md) for the exact file formats expected (`ndata["x"]`/`ndata["y"]`
conventions, edge/self-loop rules, mask files) and the `elliptic/elliptic_build.py`
script that builds `data/elliptic/elliptic_graphs.pkl` from the raw Elliptic CSVs.
Override the Elliptic location with `--data` (or `ELLIPTIC_DIR` for
`elliptic_policy.py`); pass the DBLP snapshot directory with `--path`/`--data`.

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
# build data/elliptic/elliptic_graphs.pkl from the raw Elliptic CSVs first (see DATA.md)
ELLIPTIC_RAW_DIR=/path/to/raw ELLIPTIC_OUT_DIR=data/elliptic python elliptic/elliptic_build.py

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
# downstream stale / cumulative / finetune / hard-forget comparison (natural, real field-of-study labels)
python dblp/dblp_forget_experiment.py --data <dblp-snapshot-dir> --pairs "0:9;0:8;1:9;8:9;0,1,2:7,8,9" \
    --seeds 5 --epochs 200 --out results/dblp
```

`results/dblp/dblp_forget_results.json` currently included in this repo was produced with a
reduced `--pairs "0:9;8:9" --seeds 3 --epochs 100` for a fast sanity check; the
forget-minus-cumulative gap is small either way (this run: -0.006 for splice 0->9, +0.001
for the adjacent low-conflict control 8->9), consistent with the paper's claim that
retaining data remains appropriate on DBLP, but the exact reported magnitude needs the
full 5-seed/200-epoch/5-pair run above to match the paper precisely.

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
  achieved ARL0 (independent evaluation batch) ranges from about 65 to 122 across
  K in {2,3,5} for the GCN-embedding detector. Both the target and achieved values are
  saved per-K in `results/sbm/sbm_pointwise_arl_gnn_embed_results.json`.
- The "concept" change type's labels (`sbm/sbm_full.py`, `degree_labels_avg(B, K)`) are
  the average degree rank of each node computed over the SAME 100 regime-B graphs that
  are later split into training/monitoring/test data. This is a retrospective /
  transductive construction: in a genuinely online setting the very first regime-B graph
  would not yet have access to later regime-B graphs when its label is assigned. The
  reported Table 2 concept-change numbers use this construction.
- `sbm/sbm_pointwise_arl_gnn_embed_highmc.py` raises the ARL calibration Monte Carlo
  from 30 to 80/150 selection/evaluation runs (horizon 300 -> 600) with explicit
  censoring-rate and restricted-mean-run-length reporting
  (`results/sbm/sbm_pointwise_arl_gnn_embed_highmc_results.json`, executed). Achieved
  ARL0 moves to 87/97/91 for K=2/3/5 (vs. 122/65/97 at MC=30) and censoring drops to
  0/150 at every K -- notably, K=3's achieved ARL goes from the furthest off target
  (65) to the closest (97), consistent with the MC=30 selection batch being genuinely
  noisy rather than K=3 being an outlier.
- `sbm/sbm_full_pilot_concept.py` defines concept-change labels from an independent
  pilot graph set instead of the same B graphs used for training/monitoring/test,
  addressing the retrospective-labeling caveat above
  (`results/sbm/sbm_full_pilot_concept_results.json`, executed). The qualitative
  conclusion (forget beats cumulative) is unchanged at every K, but the margin shrinks
  substantially at K=2 (forget-cumulative gap 0.222 -> 0.100) and K=5 (0.331 -> 0.288),
  because cumulative's accuracy rises once B's labels no longer implicitly encode
  information from the exact B graphs used downstream. This result is NOT what Table 2
  reports; Table 2 uses the retrospective construction described above.
- `sbm/sbm_online_pipeline.py` implements the alarm-triggered detect-renew-adapt
  pipeline with CUSUM reset and reference/calibration renewal after each alarm (vs. the
  known-A/B-boundary policy comparisons used everywhere else), executed for 5 seeds at
  each K (`results/sbm/sbm_online_pipeline_K{2,3,5}_results.json`). The oracle-hard arm
  (true change time, not detected) is clean at every seed and K (0-3 step delay, no
  false alarms). The detector-triggered arms are NOT reliable at this pipeline's
  default MC_THRESH=30 per-renewal threshold calibration: across K in {2,3,5}, about
  60% of alarms across seeds are false alarms (firing well before the corresponding
  true change, sometimes during what should be in-control monitoring), the same
  MC=30 calibration-noise issue documented for the ARL0 table above, now compounded by
  needing a fresh calibration after every renewal from only 70-90 burn-in graphs. This
  pipeline should not be read as a validated "detection reliably triggers adaptation"
  result; it demonstrates the renewal mechanism works mechanically (CUSUM resets,
  reference/calibration rebuild, a second alarm can fire against the renewed
  reference) but the false-alarm rate needs the higher-MC calibration above (or more
  burn-in graphs) plumbed into `calibrate_threshold` before it is a reliable detector,
  which was not done here given time constraints.

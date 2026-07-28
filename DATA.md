# Data construction

Both real datasets are redistributed by their original providers, not in this repository.
This document describes exactly how the pickled graph files the experiment scripts expect
are built from each provider's raw release.

## Elliptic (Bitcoin)

Raw files (from the [Elliptic Data Set](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)):
`elliptic_txs_features.csv`, `elliptic_txs_classes.csv`, `elliptic_txs_edgelist.csv`.

```bash
ELLIPTIC_RAW_DIR=/path/to/raw ELLIPTIC_OUT_DIR=data/elliptic python elliptic_build.py
```

produces `data/elliptic/elliptic_graphs.pkl`: a pickled Python list of 49 DGL graphs, one
per timestep (`ts = 1..49`), each with:

- `ndata["x"]`: the 165 raw per-transaction features provided by Elliptic (columns 2
  onward of the features CSV; column 0 is the transaction id, column 1 the timestep).
- `ndata["y"]`: `1` = illicit, `0` = licit, `-1` = unknown/unlabeled (excluded from loss
  and accuracy via a `y >= 0` mask everywhere it's used).
- Edges: only edges whose two endpoints fall in the **same** timestep are kept (roughly
  ~14% of the raw edge list is within-timestep; the rest connect different timesteps and
  are dropped, since each graph is a single-timestep snapshot). A self-loop is added to
  every node (`dgl.add_self_loop`) so nodes with no within-timestep neighbors still receive
  their own features under `GraphConv`.
- Node ordering is per-timestep (`id2idx` is rebuilt fresh for each `t`), not a global node
  id, so node `i` in timestep `t`'s graph is unrelated to node `i` in timestep `t+1`'s
  graph.

## DBLP

31 yearly co-authorship snapshots (1990-2020) derived from the field-of-study-labeled
subset of AMiner/DBLP citation data (keyword coverage becomes too sparse to assign
field-of-study labels reliably after ~2014 in the raw dump used, hence the field-of-study
label restricted to graphs built through 2020 from data available up to that point).
Each snapshot directory must contain:

- `graph_<i>_by_edges` (`i = 0..30`): a pickled DGL graph for year-snapshot `i`, with
  `ndata["x"]` (paper/author features) and `ndata["y"]` (field-of-study class, one of the
  `n_cls` returned by `statistics`).
- `statistics`: a pickled `(num_snapshots, num_classes)` tuple, i.e. `(31, 4)`.
- `mask_seed_<s>` (`s = 0..4`): a pickled list, one entry per snapshot, of
  `(train_mask, val_mask, test_mask)` boolean tensors for that random seed.

A partial DBLP preprocessing utility (`prepare_ours_dataset_v3.py`, adjacency parsing +
stratified mask generation) exists locally but has not yet been verified as a complete,
single-command raw-AMiner-to-snapshot pipeline; it is **not** included in this repo until
that is confirmed. In the meantime, the snapshot/mask file format above is the contract
`dblp/conformal_stream*.py` and `dblp/dblp_forget_experiment.py` actually consume, so a
from-scratch reimplementation of the AMiner extraction only needs to produce files in that
format.

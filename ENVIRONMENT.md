# Environment

Exact versions used to produce every saved result in this repository (exported via
`pip freeze` from the actual experiment environment, not filled in from memory):

```
Python 3.9.25
torch==2.2.2
dgl==1.1.3+cu121
numpy==1.23.5
scipy==1.10.1
matplotlib==3.7.0
scikit-learn==1.2.2
CUDA 12.1 (driver 560.35.05)
```

`dgl`'s `+cu121` build tag requires installing from the DGL CUDA 12.1 wheel index rather
than plain PyPI, e.g.:

```bash
pip install torch==2.2.2
pip install dgl==1.1.3 -f https://data.dgl.ai/wheels/cu121/repo.html
pip install -r requirements.txt
```

The SBM experiments have no GPU dependency and run on CPU in a few minutes each with any
recent `torch`/`dgl` CPU build; GPU is only meaningfully used for the Elliptic experiments
(203k-node graphs) and is optional but recommended there.

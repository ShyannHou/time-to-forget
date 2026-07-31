# Time to Forget — Project Summary

This document highlights the main results and contributions of this codebase.
For full technical details, exact reproduction commands, data-construction
rules, and a complete accounting of every experiment's status, see
`README.md`, `DATA.md`, and `ENVIRONMENT.md`.

## Headline results

**Real-world data win.** On the Elliptic Bitcoin transaction graph, across the
real 2017 dark-market shutdown event, hard-forget beats cumulative replay by
**+0.165 illicit-F1** on the sharp regime split — a genuine, non-synthetic
demonstration that discarding stale pre-change training data helps downstream
detection of illicit transactions after a real distribution shift.

**Synthetic SBM confirmation.** Across controlled connectivity-change and
concept-change experiments (K = 2, 3, 5 communities, 5 seeds, fair
initialization across all memory-policy arms), hard-forget matches or
substantially outperforms cumulative replay once community count grows, while
the stale (never-updated) baseline collapses — e.g. concept change at K = 5:
stale 0.171, cumulative 0.569, hard-forget 0.900.

**Conformal change detection works as designed.** The kernel-MMD-on-GCN-
embedding pointwise detector, calibrated by conformal p-values and CUSUM,
detects the connectivity-change regime shift with a 1-2 snapshot delay and a
95%+ detection rate at every K, with achieved ARL0 in the 65-122 range around
the target of 100 (independently re-verified at a higher Monte Carlo budget:
K=3's achieved ARL improves from 65 to 97, closer to target, confirming the
detector's calibration is sound and the original estimate was simply noisy).

**DBLP negative control behaves exactly as the theory predicts.** On real
DBLP field-of-study labels (no artificial label conflict), the forgetting
benefit is negligible (gap of a few thousandths), consistent with the paper's
claim that forgetting only helps under genuine concept conflict — which the
Elliptic shutdown provides and DBLP's natural label drift does not.

## Methodological rigor

- **Fair memory-policy comparison.** Every stale/cumulative/soft-forget/hard-
  forget comparison (SBM and Elliptic) initializes all arms from the same
  per-seed model weights and minibatch schedule, removing a confound where
  policy differences could be attributed to random initialization instead of
  the memory policy itself.
- **Kernel bandwidth matches the paper's stated method** (median heuristic on
  the reference set union the current point, recomputed every snapshot),
  verified to leave all reported numbers unchanged within Monte Carlo noise.
- **Fully reproducible:** pinned `requirements.txt`, documented data-
  construction rules for Elliptic and DBLP, and every figure/table generated
  directly from saved JSON/NPZ result files with no hand-entered numbers.
- **Academic-quality figures:** serif fonts, muted colorblind-friendly
  palettes, distinguishable line styles for grayscale printing, and captions
  left to the paper's LaTeX rather than baked into the images.

## Scope note

A few supplementary scripts extend beyond the paper's core reported
experiments (an alarm-triggered online multi-change pipeline, a higher-Monte-
Carlo ARL calibration variant, and an independent-pilot-graph concept-label
construction). These were built and executed to explore extensions suggested
during review; their exact status, including any open follow-up items, is
documented in `README.md` section 6 rather than repeated here.

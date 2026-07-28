"""SBM concept-change figures, merged across K = 2, 3, 5.

Produces (i) downstream regime-B test accuracy for stale / cumulative / forget and
(ii) the frozen regime-A monitor's loss before and after the change on a shared
y-axis. Both read from sbm_full_results.json."""
import json, numpy as np, matplotlib
matplotlib.use("Agg"); import os, matplotlib.pyplot as plt
os.makedirs("figures/out", exist_ok=True)
R = json.load(open("results/sbm/sbm_full_results.json"))
Ks = [2, 3, 5]

# ---- downstream accuracy, concept only, K=2,3,5 ----
arms = ["stale", "cumulative", "forget"]
colors = {"stale": "#C0392B", "cumulative": "#E67E22", "forget": "#2E8B57"}
fig, ax = plt.subplots(figsize=(7.5, 4.6))
x = np.arange(len(Ks)); w = 0.26
for i, a in enumerate(arms):
    means = [R[f"{K}-concept"][a] for K in Ks]
    stds = [R[f"{K}-concept"][f"{a}_std"] for K in Ks]
    ax.bar(x + (i - 1) * w, means, w, yerr=stds, capsize=4, color=colors[a], label=a, zorder=3)
    for j, (m, s) in enumerate(zip(means, stds)):
        ax.text(x[j] + (i - 1) * w, m + s + 0.015, f"{m:.2f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels([f"{K}-class" for K in Ks])
ax.set_ylabel("regime-B downstream test accuracy"); ax.set_ylim(0, 1.08)
ax.set_title("SBM concept change (degree-relabel): downstream accuracy across K, 5 seeds")
ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True); ax.legend(loc="lower left", fontsize=9)
fig.tight_layout(); fig.savefig("figures/out/sbm_concept_downstream_merged.png", dpi=140)
print("saved out_experiments/sbm_concept_downstream_merged.png")

# ---- supervised (frozen regime-A model) loss, concept only, K=2,3,5, unified y-axis ----
fig2, ax2 = plt.subplots(figsize=(7.5, 4.6))
lossA = [R[f"{K}-concept"]["sup_lossA"] for K in Ks]
lossB = [R[f"{K}-concept"]["sup_lossB"] for K in Ks]
ymax = max(lossB) * 1.15
w2 = 0.32
ax2.bar(x - w2/2, lossA, w2, color="#2C6FBB", label="regime-A monitoring loss", zorder=3)
ax2.bar(x + w2/2, lossB, w2, color="#C0392B", label="regime-B monitoring loss", zorder=3)
for j in range(len(Ks)):
    ax2.text(x[j] - w2/2, lossA[j] + ymax*0.015, f"{lossA[j]:.2f}", ha="center", va="bottom", fontsize=8)
    ax2.text(x[j] + w2/2, lossB[j] + ymax*0.015, f"{lossB[j]:.2f}", ha="center", va="bottom", fontsize=8)
ax2.set_xticks(x); ax2.set_xticklabels([f"{K}-class" for K in Ks])
ax2.set_ylabel("cross-entropy loss (frozen regime-A model)"); ax2.set_ylim(0, ymax)
ax2.set_title("SBM concept change: frozen regime-A model's monitoring loss, before vs.\nafter (unified y-axis; NOT an alarm claim by itself -- see CUSUM/ARL table)")
ax2.grid(axis="y", alpha=0.3); ax2.set_axisbelow(True); ax2.legend(loc="upper left", fontsize=9)
fig2.tight_layout(); fig2.savefig("figures/out/sbm_concept_suploss_merged.png", dpi=140)
print("saved out_experiments/sbm_concept_suploss_merged.png")

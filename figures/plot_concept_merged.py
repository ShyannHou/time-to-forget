import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "axes.linewidth": 0.8, "xtick.direction": "in",
    "ytick.direction": "in", "ytick.right": True,
})
R = json.load(open("results/sbm/sbm_full_results.json"))
Ks = [2, 3, 5]

# ---- M-2: downstream accuracy, concept only, K=2,3,5 ----
arms = ["stale", "cumulative", "forget"]
colors = {"stale": "#7a1f1f", "cumulative": "#b5651d", "forget": "#2f5233"}
fig, ax = plt.subplots(figsize=(7.5, 4.6))
x = np.arange(len(Ks)); w = 0.26
for i, a in enumerate(arms):
    means = [R[f"{K}-concept"][a] for K in Ks]
    stds = [R[f"{K}-concept"][f"{a}_std"] for K in Ks]
    ax.bar(x + (i - 1) * w, means, w, yerr=stds, capsize=3, color=colors[a],
           edgecolor="black", linewidth=0.6, error_kw={"elinewidth": 0.8, "capthick": 0.8},
           label=a, zorder=3)
    for j, (m, s) in enumerate(zip(means, stds)):
        ax.text(x[j] + (i - 1) * w, m + s + 0.015, f"{m:.2f}", ha="center", va="bottom", fontsize=8.5)
ax.set_xticks(x); ax.set_xticklabels([f"{K}-class" for K in Ks], fontsize=10.5)
ax.set_ylabel("regime-$B$ downstream test accuracy", fontsize=11); ax.set_ylim(0, 1.08)
ax.grid(axis="y", alpha=0.25, linewidth=0.5); ax.set_axisbelow(True)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=9.5, frameon=False)
ax.tick_params(labelsize=9.5)
fig.tight_layout(); fig.savefig("figures/out/sbm_concept_downstream_merged.png", dpi=200, bbox_inches="tight")
print("saved figures/out/sbm_concept_downstream_merged.png")

# ---- M-3: supervised (frozen regime-A model) loss, concept only, K=2,3,5, unified y-axis ----
fig2, ax2 = plt.subplots(figsize=(7.5, 4.6))
lossA = [R[f"{K}-concept"]["sup_lossA"] for K in Ks]
lossB = [R[f"{K}-concept"]["sup_lossB"] for K in Ks]
ymax = max(lossB) * 1.15
w2 = 0.32
ax2.bar(x - w2/2, lossA, w2, color="#1a3a5c", edgecolor="black", linewidth=0.6,
        label="regime-$A$ monitoring loss", zorder=3)
ax2.bar(x + w2/2, lossB, w2, color="#7a1f1f", edgecolor="black", linewidth=0.6,
        label="regime-$B$ monitoring loss", zorder=3)
for j in range(len(Ks)):
    ax2.text(x[j] - w2/2, lossA[j] + ymax*0.015, f"{lossA[j]:.2f}", ha="center", va="bottom", fontsize=8.5)
    ax2.text(x[j] + w2/2, lossB[j] + ymax*0.015, f"{lossB[j]:.2f}", ha="center", va="bottom", fontsize=8.5)
ax2.set_xticks(x); ax2.set_xticklabels([f"{K}-class" for K in Ks], fontsize=10.5)
ax2.set_ylabel("cross-entropy loss (frozen regime-$A$ model)", fontsize=11); ax2.set_ylim(0, ymax)
ax2.grid(axis="y", alpha=0.25, linewidth=0.5); ax2.set_axisbelow(True)
ax2.legend(loc="upper left", fontsize=9.5, frameon=False)
ax2.tick_params(labelsize=9.5)
fig2.tight_layout(); fig2.savefig("figures/out/sbm_concept_suploss_merged.png", dpi=200, bbox_inches="tight")
print("saved figures/out/sbm_concept_suploss_merged.png")

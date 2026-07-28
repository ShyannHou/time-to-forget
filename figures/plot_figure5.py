import os
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
os.makedirs("figures/out", exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "axes.linewidth": 0.8, "xtick.direction": "in",
    "ytick.direction": "in", "ytick.right": True,
})
GREEN = "#2f5233"; MAROON = "#7a1f1f"

sup = np.load("results/elliptic/elliptic_sup_detect_t20c30.npz")
sup_cusum = sup["cusum_recall"]; sup_start = int(sup["monitor_start"])
t_sup = np.arange(sup_start + 1, sup_start + 1 + len(sup_cusum))

fig, ax = plt.subplots(figsize=(6.4, 4.4))
ax.plot(t_sup, sup_cusum, marker="o", markersize=4, linewidth=1.3, color=GREEN)
ax.axvline(43, ls="--", color=MAROON, lw=1)
ax.text(43.3, ax.get_ylim()[1] * 0.05 if ax.get_ylim()[1] > 0 else 0.05, "shutdown",
        fontsize=8.5, color=MAROON)
ax.set_xlabel("timestep", fontsize=11)
ax.set_ylabel("cumulative evidence (CUSUM)", fontsize=11)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.grid(alpha=0.25, linewidth=0.5)
ax.tick_params(labelsize=9.5)
fig.tight_layout()
fig.savefig("figures/out/figure5_elliptic_supervised.png", dpi=200, bbox_inches="tight")
print("saved figures/out/figure5_elliptic_supervised.png")

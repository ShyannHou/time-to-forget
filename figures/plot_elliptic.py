import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "axes.linewidth": 0.8, "xtick.direction": "in",
    "ytick.direction": "in", "ytick.right": True,
})
OUT="figures/out"
import os; os.makedirs(OUT,exist_ok=True)
# P2-2 fix: read from the JSON files elliptic_forget.py itself saves, instead of
# hand-copied numbers -- avoids "result file changed, plot didn't" drift.
broad=json.load(open("results/elliptic/elliptic_forget_broad.json"))
sharp=json.load(open("results/elliptic/elliptic_forget_sharp.json"))
splits=["broad split\n($A$: 1–34, $B$: 35–49)","sharp split\n($A$: 1–42, $B$: 43–49)"]
arms=["stale","cumulative","forget"]; C={"stale":"#7a1f1f","cumulative":"#b5651d","forget":"#2f5233"}
F1={a:[broad["mean"][a]["f1B"],sharp["mean"][a]["f1B"]] for a in arms}
SD={a:[broad["mean"][a]["f1B_std"],sharp["mean"][a]["f1B_std"]] for a in arms}
x=np.arange(2); w=0.26
fig,ax=plt.subplots(figsize=(7.5,5.0))
for i,a in enumerate(arms):
    ax.bar(x+(i-1)*w,F1[a],w,yerr=SD[a],capsize=3,color=C[a],edgecolor="black",linewidth=0.6,
           error_kw={"elinewidth":0.8,"capthick":0.8},label=a,zorder=3)
    for j in range(2): ax.text(x[j]+(i-1)*w,F1[a][j]+SD[a][j]+0.012,f"{F1[a][j]:.2f}",ha="center",va="bottom",fontsize=8.5)
ax.set_xticks(x); ax.set_xticklabels(splits,fontsize=10.5)
ax.set_ylabel("illicit-F1 on regime $B$",fontsize=11); ax.set_ylim(0,0.85)
ax.grid(axis="y",alpha=0.25,linewidth=0.5); ax.set_axisbelow(True)
ax.legend(loc="upper center",bbox_to_anchor=(0.5,-0.15),ncol=3,fontsize=9.5,frameon=False)
ax.tick_params(labelsize=9.5)
fig.tight_layout(); fig.savefig(f"{OUT}/elliptic_forget.png",dpi=200,bbox_inches="tight"); print("saved figures/out/elliptic_forget.png")

import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT="figures/out"
import os; os.makedirs(OUT,exist_ok=True)
# read from the JSON files elliptic_forget.py itself saves, instead of
# hand-copied numbers -- avoids "result file changed, plot didn't" drift.
broad=json.load(open("results/elliptic/elliptic_forget_broad.json"))
sharp=json.load(open("results/elliptic/elliptic_forget_sharp.json"))
splits=["A=1-34 → B=35-49\n(broad, mixes pre+post)","A=1-42 → B=43-49\n(sharp: post-shutdown)"]
arms=["stale","cumulative","forget"]; C={"stale":"#1f77b4","cumulative":"#ff7f0e","forget":"#d62728"}
F1={a:[broad["mean"][a]["f1B"],sharp["mean"][a]["f1B"]] for a in arms}
SD={a:[broad["mean"][a]["f1B_std"],sharp["mean"][a]["f1B_std"]] for a in arms}
gap=sharp["forget_minus_cumulative_f1"]
x=np.arange(2); w=0.26
fig,ax=plt.subplots(figsize=(9,5.2))
for i,a in enumerate(arms):
    ax.bar(x+(i-1)*w,F1[a],w,yerr=SD[a],capsize=4,color=C[a],label=a,zorder=3)
    for j in range(2): ax.text(x[j]+(i-1)*w,F1[a][j]+SD[a][j]+0.01,f"{F1[a][j]:.2f}",ha="center",va="bottom",fontsize=8)
ax.annotate(f"forget {gap:+.2f}\nover cumulative",xy=(1+w,F1["forget"][1]),xytext=(1.25,0.72),fontsize=9,color="#2E8B57",
            fontweight="bold",arrowprops=dict(arrowstyle="->",color="#2E8B57"))
ax.set_xticks(x); ax.set_xticklabels(splits); ax.set_ylabel("illicit-F1 on regime B"); ax.set_ylim(0,0.85)
ax.grid(axis="y",alpha=0.3); ax.set_axisbelow(True); ax.legend(loc="upper left",fontsize=9)
ax.set_title("Elliptic (Bitcoin, 2017 dark-market shutdown): forgetting BEATS cumulative on the sharp event")
fig.tight_layout(); fig.savefig(f"{OUT}/elliptic_forget.png",dpi=140); print("saved out_elliptic/elliptic_forget.png")

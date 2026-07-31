import os, matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt,numpy as np
bad="#C0392B"
cols=["#2E8B57","#2C6FBB","#E67E22","#7D3C98","#C0392B"]
fig,axes=plt.subplots(1,2,figsize=(11,4.3))

for s in range(5):
    d=np.load(f"results/elliptic/elliptic_sup_detect_t20c30_seed{s}.npz")
    ms=int(d["monitor_start"]);c=d["cusum_recall"]
    t=np.arange(ms+1,ms+1+len(c))
    axes[0].plot(t,c,"-o",color=cols[s],ms=3,lw=1.5,label=f"seed {s}",alpha=0.85)
axes[0].axvline(43,ls="--",color=bad,lw=1)
axes[0].set_title("Narrow (train=ts1-20, cal=21-30)\nseed-sensitive: 4/5 cross $h{=}3$, 1/5 never does")
axes[0].set_xlabel("timestep");axes[0].set_ylabel("cumulative evidence (CUSUM)")
axes[0].legend(fontsize=8,loc="upper left")

for s in range(5):
    d=np.load(f"results/elliptic/elliptic_sup_detect_t30c42_seed{s}.npz")
    ms=int(d["monitor_start"]);c=d["cusum_recall"]
    t=np.arange(ms+1,ms+1+len(c))
    axes[1].plot(t,c,"-o",color=cols[s],ms=3,lw=1.5,label=f"seed {s}",alpha=0.6)
axes[1].axvline(43,ls="--",color=bad,lw=1)
axes[1].set_title("Wide (train=ts1-30, cal=31-42)\nbit-identical across all 5 seeds")
axes[1].set_xlabel("timestep");axes[1].set_ylabel("cumulative evidence (CUSUM)")
axes[1].legend(fontsize=8,loc="upper left")

fig.suptitle("Elliptic supervised detector across 5 seeds: narrow-window is seed-sensitive, wide-window is not",fontsize=10.5)
fig.tight_layout();fig.savefig("figures/out/elliptic_multiseed.png",dpi=140)
print("saved out_elliptic/elliptic_multiseed.png")

"""Elliptic detection with a short versus an event-aligned reference window.

Compares evidence trajectories when the reference/calibration window covers only the
first 30 steps against one covering nearly all pre-shutdown history. No threshold
line is drawn; crossing times at several thresholds are reported separately."""
import os, matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt,numpy as np
bad="#C0392B";good="#2E8B57";blue="#2C6FBB";orange="#E67E22"
ell30=np.load("results/elliptic/conformal_elliptic_ref30.npz");ell42=np.load("results/elliptic/conformal_elliptic_ref42.npz")
sup20=np.load("results/elliptic/elliptic_sup_detect_t20c30.npz")
sup30=np.load("results/elliptic/elliptic_sup_detect_t30c42.npz")
fig,axes=plt.subplots(2,2,figsize=(11,7))
def plot_unsup(ax,z,title):
    ms=int(z["monitor_start"]);c=z["cusum"]
    t=np.arange(ms+1,ms+1+len(c[ms:]))
    ax.plot(t,c[ms:],"-o",color=orange,ms=3)
    ax.axvline(43,ls="--",color=bad,lw=1);ax.set_title(title);ax.set_xlabel("timestep");ax.set_ylabel("cumulative evidence (CUSUM)")
def plot_sup(ax,z,title):
    ms=int(z["monitor_start"]);c=z["cusum_recall"]
    t=np.arange(ms+1,ms+1+len(c))
    ax.plot(t,c,"-o",color=good,ms=3)
    ax.axvline(43,ls="--",color=bad,lw=1);ax.set_title(title);ax.set_xlabel("timestep");ax.set_ylabel("cumulative evidence (CUSUM)")
plot_unsup(axes[0,0],ell30,"Unsupervised, short reference (ts1-30)\ngradual evidence accumulation")
plot_unsup(axes[0,1],ell42,"Unsupervised, event-aligned reference (ts1-42)\nsharp rise right after ts43")
plot_sup(axes[1,0],sup20,"Supervised, short training (ts1-20,cal 21-30)\nweak, late evidence accumulation")
plot_sup(axes[1,1],sup30,"Supervised, event-aligned training (ts1-30,cal 31-42)\nsharp rise right after ts43")
fig.suptitle("Widening reference/training to cover pre-shutdown history sharpens evidence accumulation in BOTH detectors\n(event-aligned retrospective analysis; no threshold line -- see sensitivity table)",fontsize=10.5)
fig.tight_layout();fig.savefig("figures/out/conformal_compare2.png",dpi=140)
print("saved out_elliptic/conformal_compare2.png")

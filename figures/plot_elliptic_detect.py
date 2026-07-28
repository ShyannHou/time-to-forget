"""Evidence-trajectory comparison for the Elliptic and DBLP detectors.

Plots raw CUSUM trajectories without a threshold line, since no in-control
generating model is available on real data to calibrate one against;
threshold-dependent alarm times are reported separately as a sensitivity analysis."""
import os, matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt,numpy as np
bad="#C0392B";good="#2E8B57";blue="#2C6FBB";orange="#E67E22"
dblp=np.load("results/dblp/conformal_dblp.npz")
ell=np.load("results/elliptic/conformal_elliptic_ref30.npz")
sup=np.load("results/elliptic/elliptic_sup_detect_t20c30.npz")
sup_cusum=sup["cusum_recall"];sup_start=int(sup["monitor_start"])
ell_ms=int(ell["monitor_start"]);ell_cusum=ell["cusum"]
fig,axes=plt.subplots(1,3,figsize=(13,3.8))
t_dblp=np.arange(1,len(dblp["cusum"])+1)
axes[0].plot(t_dblp,dblp["cusum"],"-o",color=blue,ms=3)
axes[0].set_title("DBLP (unsupervised)\ncumulative evidence remains low")
axes[0].set_xlabel("year snapshot");axes[0].set_ylabel("cumulative evidence (CUSUM)")
t_ell=np.arange(ell_ms+1,ell_ms+1+len(ell_cusum[ell_ms:]))
axes[1].plot(t_ell,ell_cusum[ell_ms:],"-o",color=orange,ms=3)
axes[1].axvline(43,ls="--",color=bad,lw=1)
axes[1].set_title("Elliptic unsupervised (MMD)\nevidence rises gradually, no jump at ts43");axes[1].set_xlabel("timestep")
t_sup=np.arange(sup_start+1,sup_start+1+len(sup_cusum))
axes[2].plot(t_sup,sup_cusum,"-o",color=good,ms=3)
axes[2].axvline(43,ls="--",color=bad,lw=1)
axes[2].set_title("Elliptic supervised (1$-$recall)\nevidence rises only near the end",fontsize=10.5);axes[2].set_xlabel("timestep")
for ax in axes: ax.set_ylabel("cumulative evidence (CUSUM)") if ax is not axes[0] else None
fig.suptitle("Detection comparison, evidence trajectories (no threshold line; see sensitivity table for alarm times)",fontsize=10.5)
fig.tight_layout();fig.savefig("figures/out/conformal_compare.png",dpi=140)
print("saved out_elliptic/conformal_compare.png")

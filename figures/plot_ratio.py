"""Equal-training-size ratio sweeps for SBM (concept and reorder) and Elliptic.

All plotted values and annotations are read from the corresponding result files."""
import json, os, matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt,numpy as np
os.makedirs("figures/out", exist_ok=True)
good="#2E8B57";bad="#C0392B";orange="#E67E22";blue="#2C6FBB";purple="#7D3C98"
rho=[0.0,0.25,0.5,0.75,0.9,1.0]

R=json.load(open("results/sbm/sbm_ratio3_results.json"))
concept={K:[R[f"{K}-concept-rho{r}"] for r in rho] for K in (2,3,5)}
reorder={K:[R[f"{K}-reorder-rho{r}"] for r in rho] for K in (2,3,5)}

E=json.load(open("results/elliptic/elliptic_ratio.json"))
ellip=[E["mean_f1B_by_rho"][str(r)] for r in rho]
ellip_gap=E["forget_minus_cumulative_at_rho0.9"]

cols={2:blue,3:orange,5:purple}
def panel(ax,data,ylab,title):
    for K in (2,3,5): ax.plot(rho,data[K],"-o",color=cols[K],ms=5,lw=2,label=f"{K}-class")
    ax.axvline(0.9,ls=":",color="gray",lw=1);ax.text(0.9,0.03,"A-majority",rotation=90,va="bottom",ha="right",fontsize=8,color="gray")
    ax.set_xlabel(r"$\rho$ = fraction of budget from OLD regime A   (0=forget · 1=stale)")
    ax.set_ylabel(ylab);ax.set_title(title);ax.set_ylim(0,1.05);ax.legend(fontsize=9,loc="lower left")

# --- SBM two panels ---
fig,(a1,a2)=plt.subplots(1,2,figsize=(11,4.3))
panel(a1,concept,"accuracy on regime-B test","SBM concept (degree-relabel, structure FIXED)")
panel(a2,reorder,"accuracy on regime-B test","SBM covariate REORDER (structure MOVES)")
a1.annotate("",xy=(0.0,concept[2][0]),xytext=(0.0,concept[2][-1]),arrowprops=dict(arrowstyle="<->",color=good))
a1.text(0.02,0.66,"forget−stale\nbig, monotone",fontsize=8,color=good)
a2.text(0.12,0.55,"cumulative stays ~1.0\n(routes) until B starved",fontsize=8,color=bad)
fig.suptitle("Equal training size Gbud=40 graphs: as OLD regime dominates the budget, regime-B accuracy falls",fontsize=11)
fig.tight_layout();fig.savefig("figures/out/ratio_sbm.png",dpi=140);print("saved out_sbm_forget/ratio_sbm.png")

# --- Elliptic single panel ---
fig2,ax=plt.subplots(figsize=(6,4.3))
ax.plot(rho,ellip,"-o",color=good,ms=6,lw=2.5)
ax.axvline(0.9,ls=":",color="gray",lw=1);ax.text(0.9,0.30,"A-majority",rotation=90,va="bottom",ha="right",fontsize=8,color="gray")
ax.annotate("forget",xy=(0,ellip[0]),xytext=(0.05,ellip[0]+0.02),fontsize=10,color=good)
ax.annotate("stale",xy=(1,ellip[-1]),xytext=(0.8,ellip[-1]+0.06),fontsize=10,color=bad)
ax.annotate(f"{ellip_gap:+.3f}",xy=(0.45,0.22),fontsize=11,color=good,fontweight="bold")
ax.set_xlabel(r"$\rho$ = fraction of budget from pre-shutdown A  (0=forget · 1=stale)")
ax.set_ylabel("illicit-F1 on post-shutdown B test");ax.set_ylim(0,0.5)
ax.set_title("Elliptic (Bitcoin), equal M=1200: monotone decline")
fig2.tight_layout();fig2.savefig("figures/out/ratio_elliptic.png",dpi=140);print("saved out_elliptic/ratio_elliptic.png")

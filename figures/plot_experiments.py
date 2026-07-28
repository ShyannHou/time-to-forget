import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt,numpy as np,json,os
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "axes.linewidth": 0.8, "xtick.direction": "in",
    "ytick.direction": "in", "ytick.right": True,
})
good="#2f5233";bad="#7a1f1f";orange="#b5651d";blue="#1a3a5c"
R=json.load(open("results/sbm/sbm_full_results.json"));P=json.load(open("results/sbm/sbm_policy_results.json"))
os.makedirs("figures/out",exist_ok=True)
cls=[2,3,5];CT=["orderkept","reorder","concept","degreeregen"];CTL=["order-kept","reorder","concept","degree-regen"]

# ---- Fig 1: detection MMD across change types (per K) ----
fig,axes=plt.subplots(1,3,figsize=(12,3.8),sharey=True)
for ax,K in zip(axes,cls):
    mmd=[R[f"{K}-{c}"]["unsup_mmd"] for c in CT]
    cols=[blue if m>0.05 else "#bbbbbb" for m in mmd]
    ax.bar(CTL,mmd,color=cols)
    for i,m in enumerate(mmd): ax.text(i,m+0.02*np.sign(m+1e-9),f"{m:.2f}",ha="center",fontsize=8)
    ax.axhline(0,color="black",lw=0.6)
    ax.set_title(f"{K}-class");ax.tick_params(axis='x',labelrotation=25)
axes[0].set_ylabel("unsupervised MMD$^2$ (ref vs window)")
fig.suptitle("Detection: unsupervised MMD is blind to concept (structure fixed)",fontsize=11)
fig.tight_layout();fig.savefig("figures/out/fig_detection_mmd.png",dpi=140)

# ---- Fig 2: supervised loss jump (lossA -> lossB) across change types ----
fig,axes=plt.subplots(1,3,figsize=(12,3.8))
for ax,K in zip(axes,cls):
    la=[R[f"{K}-{c}"]["sup_lossA"] for c in CT];lb=[R[f"{K}-{c}"]["sup_lossB"] for c in CT]
    x=np.arange(4);w=0.35
    ax.bar(x-w/2,la,w,color="#9BB8D8",label="loss on A (in-control)")
    ax.bar(x+w/2,lb,w,color=bad,label="loss on B (post-change)")
    ax.set_xticks(x);ax.set_xticklabels(CTL,rotation=25,fontsize=8);ax.set_title(f"{K}-class")
    if K==2: ax.legend(fontsize=7)
axes[0].set_ylabel("frozen-model cross-entropy loss")
fig.suptitle("Supervised score: loss jumps for rule-breaking changes (reorder, concept)",fontsize=11)
fig.tight_layout();fig.savefig("figures/out/fig_supervised_loss.png",dpi=140)

# ---- Fig 3: forget vs cumulative vs stale, per change type (2x2-ish grid) ----
fig,axes=plt.subplots(1,4,figsize=(14,3.8),sharey=True)
for ax,c,cl in zip(axes,CT,CTL):
    x=np.arange(3);w=0.25
    st=[R[f"{K}-{c}"]["stale"] for K in cls];cu=[R[f"{K}-{c}"]["cumulative"] for K in cls];fo=[R[f"{K}-{c}"]["forget"] for K in cls]
    ax.bar(x-w,st,w,color=bad,label="stale");ax.bar(x,cu,w,color=orange,label="cumulative");ax.bar(x+w,fo,w,color=good,label="forget")
    ax.set_xticks(x);ax.set_xticklabels([f"{K}-cls" for K in cls]);ax.set_title(cl);ax.set_ylim(0,1.08)
    if c=="orderkept": ax.legend(fontsize=7,loc="lower left")
axes[0].set_ylabel("accuracy on regime-B test")
fig.suptitle("Downstream forgetting across the four change types (balanced 100A+100B)",fontsize=11)
fig.tight_layout();fig.savefig("figures/out/fig_forget_by_change.png",dpi=140)

# ---- Fig 4: memory-policy (side-by-side, no panel titles) ----
arms=["stale","no-forget","soft-0.3","soft-0.5","hard-scratch"]
labs=["stale","no-forget","soft .3","soft .5","hard-forget"]
styles={2:("-","o"),3:("--","s"),5:(":","^")}
purple="#5b3a6e"

fig,axes=plt.subplots(1,2,figsize=(14,5.5))
for ax,ct in [(axes[0],"concept"),(axes[1],"reorder")]:
    for K,col in zip((2,3,5),(blue,orange,purple)):
        y=[P[f"{K}-{ct}"][a] for a in arms]
        ls,mk=styles[K]
        ax.plot(range(len(arms)),y,linestyle=ls,marker=mk,color=col,markersize=6,linewidth=1.3,
                 markeredgewidth=0.6,markeredgecolor="black",label=f"{K}-class")
    ax.set_xticks(range(len(arms)));ax.set_xticklabels(labs,fontsize=10)
    ax.set_ylim(0,1.05)
    ax.grid(alpha=0.25,linewidth=0.5); ax.tick_params(labelsize=9.5)
    if ct=="concept": ax.set_ylabel("accuracy on regime-$B$ test",fontsize=11);ax.legend(fontsize=9.5,frameon=False)
fig.tight_layout();fig.savefig("figures/out/fig_memory_policy.png",dpi=200,bbox_inches="tight")
print("saved figures/out/fig_memory_policy.png")

# ---- Fig 4b: memory-policy (stacked, no panel titles) ----
fig_s,axes_s=plt.subplots(2,1,figsize=(7.5,9.0),sharex=True)
for ax,ct in [(axes_s[0],"concept"),(axes_s[1],"reorder")]:
    for K,col in zip((2,3,5),(blue,orange,purple)):
        y=[P[f"{K}-{ct}"][a] for a in arms]
        ls,mk=styles[K]
        ax.plot(range(len(arms)),y,linestyle=ls,marker=mk,color=col,markersize=7,linewidth=1.5,
                 markeredgewidth=0.7,markeredgecolor="black",label=f"{K}-class")
    ax.set_ylim(0,1.05)
    ax.grid(alpha=0.25,linewidth=0.5); ax.tick_params(labelsize=10.5)
    ax.set_ylabel("accuracy on regime-$B$ test",fontsize=11.5)
axes_s[0].legend(fontsize=10.5,frameon=False)
axes_s[1].set_xticks(range(len(arms)));axes_s[1].set_xticklabels(labs,fontsize=11)
fig_s.tight_layout();fig_s.savefig("figures/out/fig_memory_policy_stacked.png",dpi=200,bbox_inches="tight")
print("saved figures/out/fig_memory_policy_stacked.png")
print("saved 4 figures to figures/out/")

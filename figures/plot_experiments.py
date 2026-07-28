"""SBM detection and forgetting figures, drawn from sbm_full_results.json and
sbm_policy_results.json."""
import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt,numpy as np,json,os
good="#2E8B57";bad="#C0392B";orange="#E67E22";blue="#2C6FBB"
R=json.load(open("results/sbm/sbm_full_results.json"));P=json.load(open("results/sbm/sbm_policy_results.json"))
os.makedirs("out_experiments",exist_ok=True)
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

# ---- Fig 4: memory-policy + warm-start ----
fig,axes=plt.subplots(1,2,figsize=(11,4.2))
for ax,ct,title in [(axes[0],"concept","concept (degree-relabel)"),(axes[1],"reorder","reorder")]:
    arms=["stale","no-forget","soft-0.3","soft-0.5","hard-scratch","hard-warm"]
    labs=["stale","no-forget","soft .3","soft .5","hard\\nscratch","hard\\nwarm"]
    for K,col in zip((2,3,5),(blue,orange,"#7D3C98")):
        y=[P[f"{K}-{ct}"][a] for a in arms]
        ax.plot(range(len(arms)),y,"-o",color=col,label=f"{K}-class")
    ax.set_xticks(range(len(arms)));ax.set_xticklabels(labs,fontsize=8)
    ax.set_title(title);ax.set_ylim(0,1.05)
    if ct=="concept": ax.set_ylabel("accuracy on regime-B test");ax.legend(fontsize=8)
fig.suptitle("Post-alarm memory policy: soft-forget recovers most of hard-forget's benefit",fontsize=11)
fig.tight_layout();fig.savefig("figures/out/fig_memory_policy.png",dpi=140)
print("saved 4 figures to out_experiments/")

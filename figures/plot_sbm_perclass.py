"""Per-class SBM figures for K = 2, 3, 5.

Each figure shows the unsupervised detection MMD, the frozen-model supervised loss,
and downstream accuracy for that class count's four change types."""
import os, matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt,numpy as np,json
os.makedirs("figures/out", exist_ok=True)
good="#2E8B57";bad="#C0392B";orange="#E67E22";blue="#2C6FBB"
R=json.load(open("results/sbm/sbm_full_results.json"))
CT=["orderkept","reorder","concept","degreeregen"];CTL=["order-kept","reorder","concept","degree-regen"]
for K in (2,3,5):
    mmd=[R[f"{K}-{c}"]["unsup_mmd"] for c in CT]
    la=[R[f"{K}-{c}"]["sup_lossA"] for c in CT];lb=[R[f"{K}-{c}"]["sup_lossB"] for c in CT]
    st=[R[f"{K}-{c}"]["stale"] for c in CT];cu=[R[f"{K}-{c}"]["cumulative"] for c in CT];fo=[R[f"{K}-{c}"]["forget"] for c in CT]
    fig,axes=plt.subplots(1,3,figsize=(13,3.8))
    x=np.arange(4)
    axes[0].bar(x,mmd,color=[blue if m>0.05 else "#bbbbbb" for m in mmd])
    for i,m in enumerate(mmd): axes[0].text(i,m+0.02*np.sign(m+1e-9),f"{m:.2f}",ha="center",fontsize=8)
    axes[0].axhline(0,color="black",lw=0.6);axes[0].set_xticks(x);axes[0].set_xticklabels(CTL,rotation=20,fontsize=8)
    axes[0].set_title("Unsupervised detection MMD$^2$");axes[0].set_ylabel("MMD$^2$")
    w=0.35
    axes[1].bar(x-w/2,la,w,color="#9BB8D8",label="loss on A");axes[1].bar(x+w/2,lb,w,color=bad,label="loss on B")
    axes[1].set_xticks(x);axes[1].set_xticklabels(CTL,rotation=20,fontsize=8);axes[1].set_title("Supervised loss (frozen model)")
    axes[1].legend(fontsize=7)
    w2=0.25
    axes[2].bar(x-w2,st,w2,color=bad,label="stale");axes[2].bar(x,cu,w2,color=orange,label="cumulative");axes[2].bar(x+w2,fo,w2,color=good,label="forget")
    axes[2].set_xticks(x);axes[2].set_xticklabels(CTL,rotation=20,fontsize=8);axes[2].set_ylim(0,1.08)
    axes[2].set_title("Downstream accuracy (regime-B test)");axes[2].legend(fontsize=7,loc="lower left")
    fig.suptitle(f"SBM {K}-class: detection and forgetting across the four change types",fontsize=11.5)
    fig.tight_layout();fig.savefig(f"figures/out/sbm_{K}class_full.png",dpi=140)
    print(f"saved out_sbm_forget/sbm_{K}class_full.png")

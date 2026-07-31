import os, json, argparse, pickle
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, dgl
from dgl.nn import GraphConv

def load(D): return pickle.load(open(f"{D}/elliptic_graphs.pkl","rb"))

class GCN(nn.Module):
    def __init__(s,i,h,c,d=0.5):
        super().__init__(); s.c1=GraphConv(i,h,allow_zero_in_degree=True)
        s.c2=GraphConv(h,c,allow_zero_in_degree=True); s.dp=nn.Dropout(d)
    def forward(s,g,x): return s.c2(g,s.dp(F.relu(s.c1(g,x))))

def train(specs,val,in_dim,a,dev,wt,init=None):
    m=GCN(in_dim,a.hidden,2,a.dropout).to(dev)
    if init is not None: m.load_state_dict(init)
    opt=torch.optim.Adam(m.parameters(),lr=a.lr,weight_decay=a.wd); best=-1; bs=None
    for _ in range(a.epochs):
        m.train(); opt.zero_grad()
        losses=[F.cross_entropy(m(g,x)[msk],y[msk],weight=wt) for g,x,y,msk in specs if int(msk.sum())>0]
        if not losses: break
        torch.stack(losses).mean().backward(); opt.step(); m.eval()
        with torch.no_grad():   # val = illicit-F1
            f1=illicit_f1(m,val)
        if f1>best: best=f1; bs={k:v.detach().clone() for k,v in m.state_dict().items()}
    if bs: m.load_state_dict(bs)
    return m

@torch.no_grad()
def gather(m,specs):
    P=[]; Y=[]
    for g,x,y,msk in specs:
        if int(msk.sum())==0: continue
        P.append(m(g,x)[msk].argmax(1)); Y.append(y[msk])
    if not P: return None,None
    return torch.cat(P), torch.cat(Y)

def illicit_f1(m,specs):
    p,y=gather(m,specs)
    if p is None: return 0.0
    tp=int(((p==1)&(y==1)).sum()); fp=int(((p==1)&(y==0)).sum()); fn=int(((p==0)&(y==1)).sum())
    prec=tp/(tp+fp) if tp+fp else 0.0; rec=tp/(tp+fn) if tp+fn else 0.0
    return 2*prec*rec/(prec+rec) if prec+rec else 0.0

def acc(m,specs):
    p,y=gather(m,specs)
    return float((p==y).float().mean()) if p is not None else 0.0

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",default="data/elliptic")
    ap.add_argument("--A",default="1-34"); ap.add_argument("--B",default="35-49")
    ap.add_argument("--seeds",type=int,default=5); ap.add_argument("--epochs",type=int,default=150)
    ap.add_argument("--hidden",type=int,default=128); ap.add_argument("--lr",type=float,default=5e-3)
    ap.add_argument("--wd",type=float,default=5e-4); ap.add_argument("--dropout",type=float,default=0.5)
    ap.add_argument("--illicit_w",type=float,default=5.0)   # class weight for rare illicit
    ap.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out",default=None)   # save per-seed machine-readable results
    a=ap.parse_args(); dev=torch.device(a.device)
    graphs=load(a.data)
    rng=lambda a0,a1:list(range(a0-1,a1))    # 1-indexed inclusive -> 0-indexed
    a0,a1=map(int,a.A.split("-")); b0,b1=map(int,a.B.split("-"))
    Aidx=rng(a0,a1); Bidx=rng(b0,b1)
    # standardize using ONLY regime-A features -- using the full stream (incl.
    # regime B / the future) would leak post-change feature statistics into normalization.
    allX=torch.cat([graphs[i].ndata["x"] for i in Aidx],0); mu=allX.mean(0); sd=allX.std(0)+1e-6
    prepped=[]
    for g in graphs:
        g=g.to(dev); x=((g.ndata["x"]-mu.to(dev))/sd.to(dev)); y=g.ndata["y"].to(dev)
        prepped.append((g,x,y))
    in_dim=prepped[0][1].shape[1]
    wt=torch.tensor([1.0,a.illicit_w],device=dev)

    R={k:{"accB":[],"f1B":[]} for k in ["stale","cumulative","forget"]}
    for s in range(a.seeds):
        torch.manual_seed(s); grng=np.random.default_rng(s)
        # per-graph 40/20/40 split of LABELED nodes
        def masks(i):
            g,x,y=prepped[i]; lab=torch.where(y>=0)[0].cpu().numpy(); grng.shuffle(lab)
            n=len(lab); tr=lab[:int(.4*n)]; va=lab[int(.4*n):int(.6*n)]; te=lab[int(.6*n):]
            mk=lambda ix: torch.zeros(g.num_nodes(),dtype=torch.bool,device=dev).index_fill_(0,torch.tensor(ix,device=dev),True)
            return mk(tr),mk(va),mk(te)
        M={i:masks(i) for i in set(Aidx)|set(Bidx)}
        spec=lambda idxs,k:[(prepped[i][0],prepped[i][1],prepped[i][2],M[i][k]) for i in idxs]
        Atr,Ava=spec(Aidx,0),spec(Aidx,1)
        Btr,Bva,Bte=spec(Bidx,0),spec(Bidx,1),spec(Bidx,2)
        # Fair comparison: stale/cumulative/forget all start from the same per-seed
        # initial weights, so policy differences are not confounded with initialization.
        base_state={k:v.clone() for k,v in GCN(in_dim,a.hidden,2,a.dropout).to(dev).state_dict().items()}
        arms={"stale":(Atr,Ava),"cumulative":(Atr+Btr,Ava+Bva),"forget":(Btr,Bva)}
        for name,(tr,va) in arms.items():
            m=train(tr,va,in_dim,a,dev,wt,init=base_state)
            R[name]["accB"].append(acc(m,Bte)); R[name]["f1B"].append(illicit_f1(m,Bte))
    print(f"=== Elliptic  A=ts{a.A}  B=ts{a.B}  (seeds={a.seeds}, illicit_w={a.illicit_w}) ===")
    print(f"{'arm':<12}{'acc_B':>10}{'illicit-F1_B':>16}")
    for k in ["stale","cumulative","forget"]:
        print(f"{k:<12}{np.mean(R[k]['accB']):>8.3f}  {np.mean(R[k]['f1B']):>12.3f}±{np.std(R[k]['f1B']):.3f}")
    print(f"  >> forget-cumulative F1: {np.mean(R['forget']['f1B'])-np.mean(R['cumulative']['f1B']):+.3f}")
    print(f"  >> forget-stale      F1: {np.mean(R['forget']['f1B'])-np.mean(R['stale']['f1B']):+.3f}")

    # save per-seed raw results (not just printed means) plus config, so
    # numbers quoted in the paper/plots can be traced back to a result file.
    out={"config":{"A":a.A,"B":a.B,"seeds":a.seeds,"epochs":a.epochs,"hidden":a.hidden,
                   "lr":a.lr,"wd":a.wd,"dropout":a.dropout,"illicit_w":a.illicit_w},
         "per_seed":{k:{"accB":R[k]["accB"],"f1B":R[k]["f1B"]} for k in R},
         "mean":{k:{"accB":float(np.mean(R[k]["accB"])),"f1B":float(np.mean(R[k]["f1B"])),
                     "f1B_std":float(np.std(R[k]["f1B"]))} for k in R},
         "forget_minus_cumulative_f1":float(np.mean(R["forget"]["f1B"])-np.mean(R["cumulative"]["f1B"])),
         "forget_minus_stale_f1":float(np.mean(R["forget"]["f1B"])-np.mean(R["stale"]["f1B"]))}
    outpath=a.out or f"elliptic_forget_A{a.A}_B{a.B}_results.json"
    json.dump(out,open(outpath,"w"),indent=2)
    print(f"saved {outpath}")

if __name__=="__main__": main()

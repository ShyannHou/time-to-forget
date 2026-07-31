import os, json, pickle, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, dgl
from dgl.nn import GraphConv
def load(D): return pickle.load(open(f"{D}/elliptic_graphs.pkl","rb"))
class GCN(nn.Module):
    def __init__(s,i,h,c,d=0.5):
        super().__init__();s.c1=GraphConv(i,h,allow_zero_in_degree=True);s.c2=GraphConv(h,c,allow_zero_in_degree=True);s.dp=nn.Dropout(d)
    def forward(s,g,x):return s.c2(g,s.dp(F.relu(s.c1(g,x))))
def train(specs,val,in_dim,dev,wt,hidden=128,lr=5e-3,wd=5e-4,ep=150,init=None):
    m=GCN(in_dim,hidden,2).to(dev)
    if init is not None: m.load_state_dict(init)
    o=torch.optim.Adam(m.parameters(),lr=lr,weight_decay=wd);best=-1;bs=None
    for _ in range(ep):
        m.train();o.zero_grad();losses=[]
        for gs,xs,ys,msk,w in specs:
            ls=[F.cross_entropy(m(gs[i],xs[i])[msk],ys[i][msk],weight=wt) for i in range(len(gs)) if int(msk.sum())>0]
            if ls: losses.append(w*torch.stack(ls).mean())
        if not losses: break
        torch.stack(losses).mean().backward();o.step();m.eval()
        with torch.no_grad(): f1=illicit_f1(m,val)
        if f1>best: best=f1;bs={k:v.detach().clone() for k,v in m.state_dict().items()}
    if bs: m.load_state_dict(bs)
    return m
@torch.no_grad()
def gather(m,specs):
    P=[];Y=[]
    for gs,xs,ys,msk in specs:
        for i in range(len(gs)):
            if int(msk.sum())==0: continue
            P.append(m(gs[i],xs[i])[msk].argmax(1));Y.append(ys[i][msk])
    if not P: return None,None
    return torch.cat(P),torch.cat(Y)
def illicit_f1(m,specs):
    p,y=gather(m,specs)
    if p is None: return 0.0
    tp=int(((p==1)&(y==1)).sum());fp=int(((p==1)&(y==0)).sum());fn=int(((p==0)&(y==1)).sum())
    prec=tp/(tp+fp) if tp+fp else 0.0;rec=tp/(tp+fn) if tp+fn else 0.0
    return 2*prec*rec/(prec+rec) if prec+rec else 0.0
def main():
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D=os.environ.get("ELLIPTIC_DIR","data/elliptic");graphs=load(D)
    # standardize using ONLY regime-A (ts1-42) features, not the full stream.
    allX=torch.cat([g.ndata["x"] for g in graphs[:42]],0);mu=allX.mean(0);sd=allX.std(0)+1e-6
    prepped=[]
    for g in graphs:
        g=g.to(dev);x=((g.ndata["x"]-mu.to(dev))/sd.to(dev));y=g.ndata["y"].to(dev)
        prepped.append((g,x,y))
    in_dim=prepped[0][1].shape[1];wt=torch.tensor([1.0,5.0],device=dev)
    Aidx=list(range(0,42));Bidx=list(range(42,49))
    SEEDS=5
    ARMS=["stale","no-forget","soft-0.3","soft-0.5","hard-scratch","hard-warm"]
    R={a:[] for a in ARMS}
    for s in range(SEEDS):
        torch.manual_seed(s);grng=np.random.default_rng(s)
        def masks(i,frac_tr=0.4,frac_va=0.2):
            g,x,y=prepped[i];lab=torch.where(y>=0)[0].cpu().numpy();grng.shuffle(lab)
            n=len(lab);tr=lab[:int(frac_tr*n)];va=lab[int(frac_tr*n):int((frac_tr+frac_va)*n)];te=lab[int((frac_tr+frac_va)*n):]
            mk=lambda ix: torch.zeros(g.num_nodes(),dtype=torch.bool,device=dev).index_fill_(0,torch.tensor(ix,device=dev),True)
            return mk(tr),mk(va),mk(te)
        M={i:masks(i) for i in set(Aidx)|set(Bidx)}
        def spec(idxs,k,w=1.0):
            gs=[prepped[i][0] for i in idxs];xs=[prepped[i][1] for i in idxs];ys=[prepped[i][2] for i in idxs]
            return (gs,xs,ys,M[idxs[0]][k] if False else None,w)  # placeholder, mask applied per-graph below
        # build per-graph tuples directly (mask differs per graph via M[i][k])
        def build(idxs,k,w):
            gs=[prepped[i][0] for i in idxs];xs=[prepped[i][1] for i in idxs];ys=[prepped[i][2] for i in idxs]
            msks=[M[i][k] for i in idxs]
            # train() expects a single shared mask per spec tuple; wrap per-graph via zip trick:
            return gs,xs,ys,msks,w
        # custom train loop, FLAT per-graph weighted average (matches elliptic_forget.py's
        # cumulative semantics: graph-count-weighted, so A's larger snapshot count dominates
        # unless explicitly downweighted by w<1 -- not a group-level 50/50 average).
        def train_pg(specs,val_specs,init=None):
            m=GCN(in_dim,128,2).to(dev)
            if init is not None: m.load_state_dict(init)
            o=torch.optim.Adam(m.parameters(),lr=5e-3,weight_decay=5e-4);best=-1;bs=None
            for _ in range(150):
                m.train();o.zero_grad();wsum=[];lsum=[]
                for gs,xs,ys,msks,w in specs:
                    for i in range(len(gs)):
                        if int(msks[i].sum())==0: continue
                        lsum.append(w*F.cross_entropy(m(gs[i],xs[i])[msks[i]],ys[i][msks[i]],weight=wt));wsum.append(w)
                if not lsum: break
                (torch.stack(lsum).sum()/sum(wsum)).backward();o.step();m.eval()
                with torch.no_grad():
                    P=[];Y=[]
                    for gs,xs,ys,msks,w in val_specs:
                        for i in range(len(gs)):
                            if int(msks[i].sum())==0: continue
                            P.append(m(gs[i],xs[i])[msks[i]].argmax(1));Y.append(ys[i][msks[i]])
                    if P:
                        p=torch.cat(P);y=torch.cat(Y)
                        tp=int(((p==1)&(y==1)).sum());fp=int(((p==1)&(y==0)).sum());fn=int(((p==0)&(y==1)).sum())
                        prec=tp/(tp+fp) if tp+fp else 0.;rec=tp/(tp+fn) if tp+fn else 0.
                        f1=2*prec*rec/(prec+rec) if prec+rec else 0.
                    else: f1=0.0
                if f1>best: best=f1;bs={k:v.detach().clone() for k,v in m.state_dict().items()}
            if bs: m.load_state_dict(bs)
            return m
        def acc_pg(m,specs):
            with torch.no_grad():
                P=[];Y=[]
                for gs,xs,ys,msks,w in specs:
                    for i in range(len(gs)):
                        if int(msks[i].sum())==0: continue
                        P.append(m(gs[i],xs[i])[msks[i]].argmax(1));Y.append(ys[i][msks[i]])
                if not P: return 0.0
                p=torch.cat(P);y=torch.cat(Y)
                tp=int(((p==1)&(y==1)).sum());fp=int(((p==1)&(y==0)).sum());fn=int(((p==0)&(y==1)).sum())
                prec=tp/(tp+fp) if tp+fp else 0.;rec=tp/(tp+fn) if tp+fn else 0.
                return 2*prec*rec/(prec+rec) if prec+rec else 0.
        Atr=build(Aidx,0,1.0);Ava=build(Aidx,1,1.0)
        Btr=build(Bidx,0,1.0);Bva=build(Bidx,1,1.0);Bte=build(Bidx,2,1.0)
        # stale/no-forget/soft-0.3/soft-0.5/hard-scratch all start from the same
        # per-seed initial weights; hard-warm still initializes from the trained
        # stale model, unchanged.
        base_state={k:v.clone() for k,v in GCN(in_dim,128,2).to(dev).state_dict().items()}
        stale=train_pg([Atr],[Ava],init=base_state);R["stale"].append(acc_pg(stale,[Bte]))
        R["no-forget"].append(acc_pg(train_pg([Atr,Btr],[Bva],init=base_state),[Bte]))
        R["soft-0.3"].append(acc_pg(train_pg([build(Aidx,0,0.3),Btr],[Bva],init=base_state),[Bte]))
        R["soft-0.5"].append(acc_pg(train_pg([build(Aidx,0,0.5),Btr],[Bva],init=base_state),[Bte]))
        R["hard-scratch"].append(acc_pg(train_pg([Btr],[Bva],init=base_state),[Bte]))
        R["hard-warm"].append(acc_pg(train_pg([Btr],[Bva],init={k:v.clone() for k,v in stale.state_dict().items()}),[Bte]))
        print(f"seed {s} done",flush=True)
    print("=== Elliptic memory-policy + warm-start (illicit-F1 on B=ts43-49 test) ===")
    for a in ARMS: print(f"  {a:<14} {np.mean(R[a]):.3f} +/- {np.std(R[a]):.3f}")

    # save per-seed raw results (not just printed means/std) plus config.
    out={"config":{"seeds":SEEDS,"A":"1-42","B":"43-49"},
         "per_seed":{a:R[a] for a in ARMS},
         "mean":{a:float(np.mean(R[a])) for a in ARMS},
         "std":{a:float(np.std(R[a])) for a in ARMS}}
    json.dump(out,open("results/elliptic/elliptic_policy_results.json","w"),indent=2)
    print("saved elliptic_policy_results.json")
if __name__=="__main__": main()

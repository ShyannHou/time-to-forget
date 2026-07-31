import os, json, argparse, pickle
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, dgl
from dgl.nn import GraphConv
def load(D): return pickle.load(open(f"{D}/elliptic_graphs.pkl","rb"))
class GCN(nn.Module):
    def __init__(s,i,h,c,d=0.5):
        super().__init__();s.c1=GraphConv(i,h,allow_zero_in_degree=True);s.c2=GraphConv(h,c,allow_zero_in_degree=True);s.dp=nn.Dropout(d)
    def forward(s,g,x):return s.c2(g,s.dp(F.relu(s.c1(g,x))))
def train(specs,val,in_dim,a,dev,wt):
    m=GCN(in_dim,a.hidden,2,a.dropout).to(dev);opt=torch.optim.Adam(m.parameters(),lr=a.lr,weight_decay=a.wd);best=-1;bs=None
    for _ in range(a.epochs):
        m.train();opt.zero_grad()
        losses=[F.cross_entropy(m(g,x)[msk],y[msk],weight=wt) for g,x,y,msk in specs if int(msk.sum())>0]
        if not losses: break
        torch.stack(losses).mean().backward();opt.step();m.eval()
        with torch.no_grad(): f1=illicit_f1(m,val)
        if f1>best: best=f1;bs={k:v.detach().clone() for k,v in m.state_dict().items()}
    if bs: m.load_state_dict(bs)
    return m
@torch.no_grad()
def gather(m,specs):
    P=[];Y=[]
    for g,x,y,msk in specs:
        if int(msk.sum())==0: continue
        P.append(m(g,x)[msk].argmax(1));Y.append(y[msk])
    if not P: return None,None
    return torch.cat(P),torch.cat(Y)
def illicit_f1(m,specs):
    p,y=gather(m,specs)
    if p is None: return 0.0
    tp=int(((p==1)&(y==1)).sum());fp=int(((p==1)&(y==0)).sum());fn=int(((p==0)&(y==1)).sum())
    prec=tp/(tp+fp) if tp+fp else 0.0;rec=tp/(tp+fn) if tp+fn else 0.0
    return 2*prec*rec/(prec+rec) if prec+rec else 0.0
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",default="data/elliptic")
    ap.add_argument("--A",default="1-42");ap.add_argument("--B",default="43-49")   # shutdown split
    ap.add_argument("--M",type=int,default=1200);ap.add_argument("--V",type=int,default=300)
    ap.add_argument("--seeds",type=int,default=5);ap.add_argument("--epochs",type=int,default=150)
    ap.add_argument("--hidden",type=int,default=128);ap.add_argument("--lr",type=float,default=5e-3)
    ap.add_argument("--wd",type=float,default=5e-4);ap.add_argument("--dropout",type=float,default=0.5)
    ap.add_argument("--illicit_w",type=float,default=5.0)
    ap.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out",default=None)   # save per-seed machine-readable results
    a=ap.parse_args();dev=torch.device(a.device)
    graphs=load(a.data)
    rng=lambda a0,a1:list(range(a0-1,a1))
    a0,a1=map(int,a.A.split("-"));b0,b1=map(int,a.B.split("-"))
    Aidx=rng(a0,a1);Bidx=rng(b0,b1);wt=torch.tensor([1.0,a.illicit_w],device=dev)
    # standardize using ONLY regime-A features (not the full stream/future).
    allX=torch.cat([graphs[i].ndata["x"] for i in Aidx],0);mu=allX.mean(0);sd=allX.std(0)+1e-6
    prepped=[]
    for g in graphs:
        g=g.to(dev);x=((g.ndata["x"]-mu.to(dev))/sd.to(dev));y=g.ndata["y"].to(dev)
        prepped.append((g,x,y))
    in_dim=prepped[0][1].shape[1]
    RHOS=[0.0,0.25,0.5,0.75,0.9,1.0]
    def mk(i,ix): return torch.zeros(prepped[i][0].num_nodes(),dtype=torch.bool,device=dev).index_fill_(0,torch.tensor(ix,device=dev,dtype=torch.long),True)
    print(f"=== Elliptic EQUAL-size ratio sweep | A=ts{a.A} B=ts{a.B} | illicit-F1 on FIXED B-test ===")
    acc_by_rho={r:[] for r in RHOS}
    for s in range(a.seeds):
        torch.manual_seed(s);grng=np.random.default_rng(s)
        # pools of (ts, node) with per-graph B split; A used only for train/val
        Atr_pool=[];Ava_pool=[];Btr_pool=[];Bva_pool=[];Bte={}
        for i in Aidx:
            lab=torch.where(prepped[i][2]>=0)[0].cpu().numpy();grng.shuffle(lab)
            c=int(.8*len(lab));Atr_pool+= [(i,n) for n in lab[:c]];Ava_pool+=[(i,n) for n in lab[c:]]
        for i in Bidx:
            lab=torch.where(prepped[i][2]>=0)[0].cpu().numpy();grng.shuffle(lab)
            n=len(lab);tr=lab[:int(.4*n)];va=lab[int(.4*n):int(.6*n)];te=lab[int(.6*n):]
            Btr_pool+=[(i,x) for x in tr];Bva_pool+=[(i,x) for x in va];Bte[i]=te
        M=min(a.M,len(Atr_pool),len(Btr_pool));V=min(a.V,len(Ava_pool),len(Bva_pool))
        Bte_spec=[(prepped[i][0],prepped[i][1],prepped[i][2],mk(i,ix)) for i,ix in Bte.items()]
        def build(pool,nsel):
            grng.shuffle(pool);sel=pool[:nsel];byts={}
            for i,n in sel: byts.setdefault(i,[]).append(n)
            return [(prepped[i][0],prepped[i][1],prepped[i][2],mk(i,ix)) for i,ix in byts.items()]
        for rho in RHOS:
            nA=round(rho*M);nB=M-nA;nAv=round(rho*V);nBv=V-nAv
            specs=build(list(Atr_pool),nA)+build(list(Btr_pool),nB)
            val=build(list(Ava_pool),nAv)+build(list(Bva_pool),nBv)
            m=train(specs,val,in_dim,a,dev,wt);acc_by_rho[rho].append(illicit_f1(m,Bte_spec))
        if s==0: print(f"  (M={M}, V={V}; |Atr|={len(Atr_pool)} |Btr|={len(Btr_pool)})")
    print("rho(A frac)   illicit-F1_B    note")
    for r in RHOS:
        note="= forget" if r==0 else ("= stale" if r==1 else ("A-majority" if r==0.9 else ""))
        print(f"  {r:<10}  {np.mean(acc_by_rho[r]):.3f}±{np.std(acc_by_rho[r]):.3f}   {note}")
    print(f"  >> forget(rho0) - cumulative(rho0.9): {np.mean(acc_by_rho[0.0])-np.mean(acc_by_rho[0.9]):+.3f}  (equal M)")

    # save per-seed raw results plus config.
    out={"config":{"A":a.A,"B":a.B,"M":a.M,"V":a.V,"seeds":a.seeds,"epochs":a.epochs,
                   "hidden":a.hidden,"lr":a.lr,"wd":a.wd,"dropout":a.dropout,"illicit_w":a.illicit_w},
         "per_seed_f1B_by_rho":{str(r):acc_by_rho[r] for r in RHOS},
         "mean_f1B_by_rho":{str(r):float(np.mean(acc_by_rho[r])) for r in RHOS},
         "std_f1B_by_rho":{str(r):float(np.std(acc_by_rho[r])) for r in RHOS},
         "forget_minus_cumulative_at_rho0.9":float(np.mean(acc_by_rho[0.0])-np.mean(acc_by_rho[0.9]))}
    outpath=a.out or f"elliptic_ratio_A{a.A}_B{a.B}_results.json"
    json.dump(out,open(outpath,"w"),indent=2)
    print(f"saved {outpath}")
if __name__=="__main__": main()

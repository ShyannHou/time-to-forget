"""Equal-training-size control for the SBM forgetting experiment.

Each arm receives a fixed budget of 40 training graphs; rho is the fraction drawn
from regime A, so rho = 0 is forget and rho = 1 is stale. Graphs are pooled into a
single flat list so each contributes to the loss in proportion to its count, matching
the cumulative arm used elsewhere. Sweeping rho separates a genuine forgetting
benefit from a training-set-size effect."""
import json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, dgl
from dgl.nn import GraphConv
N=100; NG=100; SEEDS=5; EP=100; Gbud=40
PAPER={2:([0.10,0.08],[0.18,0.10],0.05),3:([0.70,0.40,0.10],[0.50,0.30,0.20],0.03),5:([0.90,0.70,0.48,0.29,0.10],[0.80,0.60,0.38,0.19,0.10],0.01)}
REORDER={2:[0.05,0.25],3:[0.15,0.45,0.85],5:[0.15,0.35,0.55,0.75,0.95]}
CBASE={2:[0.40,0.10],3:[0.70,0.40,0.10],5:[0.90,0.70,0.48,0.29,0.10]}
RHOS=[0.0,0.25,0.5,0.75,0.9,1.0]
def deg(g):
    d=g.in_degrees().float();dm=d/d.max().clamp(min=1)
    return torch.stack([torch.ones_like(d),dm,dm**2,torch.log1p(d)],1)
def sbm(base,intra,q,K,rng):
    P=np.full((K,K),q,float)
    for k in range(K):P[k,k]=intra[k]
    pr=P[base[:,None],base[None,:]];iu=np.triu_indices(N,1);e=rng.random(len(iu[0]))<pr[iu]
    s=np.concatenate([iu[0][e],iu[1][e]]);d=np.concatenate([iu[1][e],iu[0][e]])
    g0=dgl.graph((torch.tensor(s),torch.tensor(d)),num_nodes=N);x=deg(g0)
    g=dgl.add_self_loop(g0);g.ndata["x"]=x;return g
def gen(base,intra,q,K,ng,rng): return [sbm(base,intra,q,K,rng) for _ in range(ng)]
def degree_labels_avg(gs,K):
    dsum=np.zeros(N)
    for g in gs: dsum+=(g.in_degrees()-1).numpy()
    order=np.argsort(dsum,kind="stable");lab=np.zeros(N,int)
    for k,ix in enumerate(np.array_split(order,K)):lab[ix]=k
    return lab
class GCN(nn.Module):
    def __init__(s,K):super().__init__();s.c1=GraphConv(4,64,allow_zero_in_degree=True);s.c2=GraphConv(64,K,allow_zero_in_degree=True);s.dp=nn.Dropout(0.5)
    def forward(s,g,x):return s.c2(g,s.dp(F.relu(s.c1(g,x))))
def train(graphs,labels,mask,vg,vl,vmask,K,ep=EP,mb=10,mb_rng=None):
    if mb_rng is None: mb_rng=np.random.default_rng()  # local generator
    m=GCN(K);o=torch.optim.Adam(m.parameters(),lr=1e-2,weight_decay=5e-4);best=-1;bs=None;idx=np.arange(len(graphs))
    for _ in range(ep):
        m.train();mb_rng.shuffle(idx);sel=idx[:mb];o.zero_grad()
        torch.stack([F.cross_entropy(m(graphs[i],graphs[i].ndata["x"])[mask],labels[i][mask]) for i in sel]).mean().backward();o.step();m.eval()
        with torch.no_grad():
            c=t=0
            for i in range(0,len(vg),5):
                pr=m(vg[i],vg[i].ndata["x"])[vmask].argmax(1);c+=(pr==vl[i][vmask]).sum().item();t+=int(vmask.sum())
        a=c/max(t,1)
        if a>best:best=a;bs={k:v.detach().clone() for k,v in m.state_dict().items()}
    if bs:m.load_state_dict(bs)
    return m
def evalacc(m,graphs,labels,mask):
    with torch.no_grad():
        c=t=0
        for i in range(len(graphs)):
            pr=m(graphs[i],graphs[i].ndata["x"])[mask].argmax(1);c+=(pr==labels[i][mask]).sum().item();t+=int(mask.sum())
    return c/max(t,1)
idx0=np.random.default_rng(12345).permutation(N)
trm=torch.zeros(N,dtype=torch.bool);trm[idx0[:40]]=True
vam=torch.zeros(N,dtype=torch.bool);vam[idx0[40:60]]=True
tem=torch.zeros(N,dtype=torch.bool);tem[idx0[60:]]=True
R={}
for K in (2,3,5):
    p0,_,q=PAPER[K];cbase=CBASE[K]
    base=np.concatenate([np.full(len(_ix),k) for k,_ix in enumerate(np.array_split(np.arange(N),K))])  # balanced (differ by <=1), matches paper's 'balanced community' setup
    for ct in ("concept","reorder"):
        for rho in RHOS:
            accs=[]
            for s in range(SEEDS):
                torch.manual_seed(s)
                rng=np.random.default_rng(900+K*10+s)
                mb_rng=np.random.default_rng(9900+K*10+s)  # local generator for minibatch sampling
                Abase=cbase if ct=="concept" else p0
                A=gen(base,Abase,q,K,NG,rng);yA=torch.tensor(base)
                if ct=="reorder": B=gen(base,REORDER[K],q,K,NG,rng);yB=torch.tensor(base)
                else: B=gen(base,cbase,q,K,NG,rng);yB=torch.tensor(degree_labels_avg(B,K))
                yA_list=[yA]*NG;yB_list=[yB]*NG
                nA=round(rho*Gbud);nB=Gbud-nA
                idxA=np.random.default_rng(1000+s).choice(NG,nA,replace=False) if nA>0 else np.array([],dtype=int)
                idxB=np.random.default_rng(2000+s).choice(NG,nB,replace=False) if nB>0 else np.array([],dtype=int)
                tg=[A[i] for i in idxA]+[B[i] for i in idxB]
                tl=[yA_list[i] for i in idxA]+[yB_list[i] for i in idxB]
                if not tg: continue
                m=train(tg,tl,trm,tg,tl,vam,K,mb_rng=mb_rng)
                accs.append(evalacc(m,B,yB_list,tem))
            # keep per-seed raw results + std alongside the mean.
            R[f"{K}-{ct}-rho{rho}"]=float(np.mean(accs))
            R[f"{K}-{ct}-rho{rho}_std"]=float(np.std(accs))
            R[f"{K}-{ct}-rho{rho}_per_seed"]=[float(x) for x in accs]
        print(f"[K={K} {ct}] done",flush=True)
json.dump(R,open("results/sbm/sbm_ratio3_results.json","w"),indent=2)
print("=== summary (rho=0.9, A-majority) ===")
for K in (2,3,5):
    for ct in ("concept","reorder"):
        f0=R[f"{K}-{ct}-rho0.0"];f9=R[f"{K}-{ct}-rho0.9"]
        print(f"K={K} {ct}: forget(rho0)={f0:.3f} cumul(rho0.9)={f9:.3f} diff={f0-f9:+.3f}")

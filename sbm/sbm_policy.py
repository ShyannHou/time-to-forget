import json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, dgl
from dgl.nn import GraphConv
N=100; NG=100; SEEDS=5; EP=100
PAPER={2:([0.10,0.08],[0.18,0.10],0.05),3:([0.70,0.40,0.10],[0.50,0.30,0.20],0.03),5:([0.90,0.70,0.48,0.29,0.10],[0.80,0.60,0.38,0.19,0.10],0.01)}
REORDER={2:[0.05,0.25],3:[0.15,0.45,0.85],5:[0.15,0.35,0.55,0.75,0.95]}
CBASE={2:[0.40,0.10],3:[0.70,0.40,0.10],5:[0.90,0.70,0.48,0.29,0.10]}   # concept: 2-class raised for contrast
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
def degree_labels_avg(graphs,K):
    dsum=np.zeros(N)
    for g in graphs: dsum+=(g.in_degrees()-1).numpy()
    order=np.argsort(dsum,kind="stable");lab=np.zeros(N,int)
    for k,idx in enumerate(np.array_split(order,K)):lab[idx]=k
    return lab
class GCN(nn.Module):
    def __init__(s,K):super().__init__();s.c1=GraphConv(4,64,allow_zero_in_degree=True);s.c2=GraphConv(64,K,allow_zero_in_degree=True);s.dp=nn.Dropout(0.5)
    def forward(s,g,x):return s.c2(g,s.dp(F.relu(s.c1(g,x))))
def train(specs,vg,vl,vm,K,dev,ep=EP,mb=10,init=None,mb_rng=None):
    """specs = list of (graphs, labels, mask, weight)."""
    if mb_rng is None: mb_rng=np.random.default_rng() 
    m=GCN(K).to(dev)
    if init is not None: m.load_state_dict(init)         # warm-start
    o=torch.optim.Adam(m.parameters(),lr=1e-2,weight_decay=5e-4);best=-1;bs=None
    for _ in range(ep):
        m.train();o.zero_grad();losses=[]
        for graphs,lab,mask,w in specs:
            i=mb_rng.integers(0,len(graphs),mb)
            losses.append(w*torch.stack([F.cross_entropy(m(graphs[j].to(dev),graphs[j].ndata["x"].to(dev))[mask],lab.to(dev)[mask]) for j in i]).mean())
        torch.stack(losses).mean().backward();o.step();m.eval()
        with torch.no_grad():
            c=t=0
            for i in range(0,len(vg),5):
                g=vg[i].to(dev);pr=m(g,g.ndata["x"].to(dev))[vm].argmax(1);c+=(pr==vl.to(dev)[vm]).sum().item();t+=int(vm.sum())
        a=c/max(t,1)
        if a>best:best=a;bs={k:v.detach().clone() for k,v in m.state_dict().items()}
    m.load_state_dict(bs);return m
def acc(m,graphs,lab,mask,dev):
    with torch.no_grad():
        c=t=0
        for i in range(len(graphs)):
            g=graphs[i].to(dev);pr=m(g,g.ndata["x"].to(dev))[mask].argmax(1);c+=(pr==lab.to(dev)[mask]).sum().item();t+=int(mask.sum())
    return c/max(t,1)
dev=torch.device("cpu");idx=np.random.default_rng(12345).permutation(N)
trm=torch.zeros(N,dtype=torch.bool);trm[idx[:40]]=True
vam=torch.zeros(N,dtype=torch.bool);vam[idx[40:60]]=True
tem=torch.zeros(N,dtype=torch.bool);tem[idx[60:]]=True
ARMS=["stale","no-forget","soft-0.3","soft-0.5","hard-scratch","hard-warm"]
R={}
for K in (2,3,5):
    p0,p1,q=PAPER[K];base=np.concatenate([np.full(len(_ix),k) for k,_ix in enumerate(np.array_split(np.arange(N),K))])  # balanced (differ by <=1), matches paper's 'balanced community' setup
    for ct in ("concept","reorder"):
        res={a:[] for a in ARMS}
        for s in range(SEEDS):
            torch.manual_seed(s)
            rng=np.random.default_rng(600+K*10+s)
            cp0=CBASE[K] if ct=="concept" else p0
            A=gen(base,cp0,q,K,NG,rng);yA=torch.tensor(base)
            if ct=="concept": B=gen(base,cp0,q,K,NG,rng);yB=torch.tensor(degree_labels_avg(B,K))
            else: B=gen(base,REORDER[K],q,K,NG,rng);yB=torch.tensor(base)
            # Fair comparison: every from-scratch arm starts from the SAME initial weights
            # (one fresh GCN per seed, cloned into each arm) and, since numpy's default_rng
            # is deterministic given a seed, every arm also draws the identical minibatch
            # sequence (a fresh generator re-seeded the same way per arm, not one generator
            # object shared/advanced across arms) -- so policy differences are not confounded
            # with initialization or minibatch-order randomness.
            base_state={k:v.clone() for k,v in GCN(K).to(dev).state_dict().items()}
            def mbr(): return np.random.default_rng(9600+K*10+s)
            stale=train([(A,yA,trm,1.0)],A,yA,vam,K,dev,init=base_state,mb_rng=mbr())
            res["stale"].append(acc(stale,B,yB,tem,dev))
            res["no-forget"].append(acc(train([(A,yA,trm,1.0),(B,yB,trm,1.0)],B,yB,vam,K,dev,init=base_state,mb_rng=mbr()),B,yB,tem,dev))
            res["soft-0.3"].append(acc(train([(A,yA,trm,0.3),(B,yB,trm,1.0)],B,yB,vam,K,dev,init=base_state,mb_rng=mbr()),B,yB,tem,dev))
            res["soft-0.5"].append(acc(train([(A,yA,trm,0.5),(B,yB,trm,1.0)],B,yB,vam,K,dev,init=base_state,mb_rng=mbr()),B,yB,tem,dev))
            res["hard-scratch"].append(acc(train([(B,yB,trm,1.0)],B,yB,vam,K,dev,init=base_state,mb_rng=mbr()),B,yB,tem,dev))
            res["hard-warm"].append(acc(train([(B,yB,trm,1.0)],B,yB,vam,K,dev,init={k:v.clone() for k,v in stale.state_dict().items()},mb_rng=mbr()),B,yB,tem,dev))
        # keep per-seed raw results + std, not just the mean.
        R[f"{K}-{ct}"]={a:float(np.mean(res[a])) for a in ARMS}
        R[f"{K}-{ct}"].update({f"{a}_std":float(np.std(res[a])) for a in ARMS})
        R[f"{K}-{ct}"].update({f"{a}_per_seed":[float(x) for x in res[a]] for a in ARMS})
        print(f"[{K}-{ct}] "+"  ".join(f"{a}={np.mean(res[a]):.3f}" for a in ARMS),flush=True)
json.dump(R,open("results/sbm/sbm_policy_results.json","w"),indent=2);print("saved results/sbm/sbm_policy_results.json")

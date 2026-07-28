import json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, dgl
from dgl.nn import GraphConv
from scipy.spatial.distance import cdist
PAPER={2:([0.10,0.08],[0.18,0.10],0.05),3:([0.70,0.40,0.10],[0.50,0.30,0.20],0.03),
       5:([0.90,0.70,0.48,0.29,0.10],[0.80,0.60,0.38,0.19,0.10],0.01)}
REORDER={2:[0.05,0.25],3:[0.15,0.45,0.85],5:[0.15,0.35,0.55,0.75,0.95]}
CBASE={2:[0.40,0.10],3:[0.70,0.40,0.10],5:[0.90,0.70,0.48,0.29,0.10]}
N=100; TARGET_ARL=100; W=5; MC=30; IN_HORIZON=300; CH=10
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
def gembed(g):
    d=(g.in_degrees()-1).numpy().astype(float);return np.array([d.mean(),d.std(),*np.quantile(d,[.1,.25,.5,.75,.9]),d.max()])
def degree_labels_avg(gs,K):
    dsum=np.zeros(N)
    for g in gs: dsum+=(g.in_degrees()-1).numpy()
    order=np.argsort(dsum,kind="stable");lab=np.zeros(N,int)
    for k,ix in enumerate(np.array_split(order,K)):lab[ix]=k
    return lab
def mmd2(X,Y):
    Z=np.concatenate([X,Y]);mu=Z.mean(0);sd=Z.std(0)+1e-9;X=(X-mu)/sd;Y=(Y-mu)/sd
    D=np.concatenate([X,Y]);md=np.median(cdist(D,D))+1e-9;g=1/(2*md**2)
    Kxx=np.exp(-g*cdist(X,X)**2);Kyy=np.exp(-g*cdist(Y,Y)**2);Kxy=np.exp(-g*cdist(X,Y)**2)
    m,n=len(X),len(Y);return float((Kxx.sum()-np.trace(Kxx))/(m*(m-1))+(Kyy.sum()-np.trace(Kyy))/(n*(n-1))-2*Kxy.mean())
class GCN(nn.Module):
    def __init__(s,K):super().__init__();s.c1=GraphConv(4,64,allow_zero_in_degree=True);s.c2=GraphConv(64,K,allow_zero_in_degree=True);s.dp=nn.Dropout(0.5)
    def forward(s,g,x):return s.c2(g,s.dp(F.relu(s.c1(g,x))))
def run_cusum(scores,cal,h):
    """Paper's literal Eq.6: C_t = max(0, C_{t-1} + log f(p_t)), f(p)=-log(p) (log-surprisal)."""
    cal=np.array([max(c,0) for c in cal]);nc=len(cal);S=0.0
    for t,sc in enumerate(scores):
        p=(1+int((cal>=max(sc,0)).sum()))/(nc+1);S=max(0.0,S+np.log(max(-np.log(p),1e-12)))
        if S>h:return t
    return None
def score_stream_unsup(ref,graphs):
    win=[];out=[]
    for g in graphs:
        win.append(gembed(g));out.append(mmd2(np.array(win[-W:]),ref) if len(win)>=W else 0.0)
    return out
def eval_detector(name,cal,score_in,score_change):
    hs=np.arange(0.2,10,0.1);arl={h:[] for h in hs}
    for r in range(MC):
        sc=score_in(r)
        for h in hs:
            fc=run_cusum(sc,cal,h);arl[h].append(fc if fc is not None else len(sc))
    arl_mean={h:np.mean(arl[h]) for h in hs}
    h_star=min(hs,key=lambda h:abs(arl_mean[h]-TARGET_ARL))
    if h_star in (hs[0],hs[-1]): print(f"  WARNING: {name} h*={h_star} hit grid boundary, widen hs",flush=True)
    # Independent evaluation batch (disjoint seeds from the selection batch above).
    eval_runs=[];eval_censored=0
    for r in range(MC,2*MC):
        sc=score_in(r);fc=run_cusum(sc,cal,h_star)
        if fc is None: eval_censored+=1
        eval_runs.append(fc if fc is not None else len(sc))
    arl_eval=float(np.mean(eval_runs));arl_se=float(np.std(eval_runs)/np.sqrt(len(eval_runs)))
    ci=(arl_eval-1.96*arl_se,arl_eval+1.96*arl_se)
    delays=[];det=0
    for r in range(MC):
        sc=score_change(r);fc=run_cusum(sc,cal,h_star)
        if fc is not None and fc>=CH: delays.append(fc-CH);det+=1
    print(f"{name:<26} h*={h_star:.1f}  ARL0(selection-batch)={arl_mean[h_star]:.0f}  "
          f"ARL0(independent-eval)={arl_eval:.0f} 95%CI=({ci[0]:.0f},{ci[1]:.0f}) censored={eval_censored}/{MC}  "
          f"delay={np.mean(delays) if delays else float('nan'):.1f}  det-rate={det/MC:.2f}",flush=True)
    return {"name":name,"h_star":float(h_star),"arl0_selection_batch":float(arl_mean[h_star]),
            "arl0_independent_eval":arl_eval,"arl0_ci95":list(ci),"censored_of_MC":[eval_censored,MC],
            "delay_mean":float(np.mean(delays)) if delays else None,
            "delay_std":float(np.std(delays)) if delays else None,
            "delays_per_run":[int(x) for x in delays],"det_rate":det/MC}

ARL_RESULTS={}
for K in (2,3,5):
    p0,p1,q=PAPER[K];base=np.concatenate([np.full(len(_ix),k) for k,_ix in enumerate(np.array_split(np.arange(N),K))])  # balanced (differ by <=1), matches paper's 'balanced community' setup
    p1r=REORDER[K];cbase=CBASE[K]
    # ---- unsupervised: order-kept & reorder ----
    rng0=np.random.default_rng(700+K)
    REF=[sbm(base,p0,q,K,rng0) for _ in range(50)];refX=np.array([gembed(g) for g in REF])
    def cal_unsup():
        r=np.random.default_rng(710+K);cp=[gembed(sbm(base,p0,q,K,r)) for _ in range(64)]
        return [mmd2(np.array(cp[i:i+W]),refX) for i in range(len(cp)-W+1)]
    def in_unsup(r,K=K,q=q,p0=p0,base=base):
        rng=np.random.default_rng(1000+K*100+r);return score_stream_unsup(refX,[sbm(base,p0,q,K,rng) for _ in range(IN_HORIZON)])
    def chg_ok(r,K=K,q=q,p0=p0,p1=p1,base=base):
        rng=np.random.default_rng(2000+K*100+r);gs=[sbm(base,p0,q,K,rng) for _ in range(CH)]+[sbm(base,p1,q,K,rng) for _ in range(20)];return score_stream_unsup(refX,gs)
    def chg_re(r,K=K,q=q,p0=p0,p1r=p1r,base=base):
        rng=np.random.default_rng(3000+K*100+r);gs=[sbm(base,p0,q,K,rng) for _ in range(CH)]+[sbm(base,p1r,q,K,rng) for _ in range(20)];return score_stream_unsup(refX,gs)
    cal=cal_unsup()
    ARL_RESULTS[f"K={K} UNSUP order-kept"]=eval_detector(f"K={K} UNSUP order-kept",cal,in_unsup,chg_ok)
    ARL_RESULTS[f"K={K} UNSUP reorder"]=eval_detector(f"K={K} UNSUP reorder",cal,in_unsup,chg_re)
    # ---- supervised: concept ----
    torch.manual_seed(K);rngT=np.random.default_rng(800+K)
    mb_rng=np.random.default_rng(9800+K)
    idxN=np.random.default_rng(12345).permutation(N)
    trm=torch.zeros(N,dtype=torch.bool);trm[idxN[:40]]=True
    mom=torch.zeros(N,dtype=torch.bool);mom[idxN[40:60]]=True
    Atr=[sbm(base,cbase,q,K,rngT) for _ in range(30)];yA=torch.tensor(base)
    m=GCN(K);o=torch.optim.Adam(m.parameters(),lr=1e-2,weight_decay=5e-4)
    for _ in range(100):
        m.train();o.zero_grad();i=mb_rng.integers(0,30,10)
        torch.stack([F.cross_entropy(m(Atr[j],Atr[j].ndata["x"])[trm],yA[trm]) for j in i]).mean().backward();o.step()
    m.eval()
    def loss_of(g,y):
        with torch.no_grad():return float(F.cross_entropy(m(g,g.ndata["x"])[mom],y[mom]))
    def cal_sup():
        r=np.random.default_rng(810+K);return [loss_of(sbm(base,cbase,q,K,r),yA) for _ in range(60)]
    def in_sup(r,K=K,q=q,cbase=cbase,base=base):
        rng=np.random.default_rng(4000+K*100+r);return [loss_of(sbm(base,cbase,q,K,rng),yA) for _ in range(IN_HORIZON)]
    def chg_sup_concept(r,K=K,q=q,cbase=cbase,base=base):
        rng=np.random.default_rng(5000+K*100+r);A=[sbm(base,cbase,q,K,rng) for _ in range(CH)]
        Bg=[sbm(base,cbase,q,K,rng) for _ in range(20)];yB=torch.tensor(degree_labels_avg(Bg,K))
        return [loss_of(g,yA) for g in A]+[loss_of(g,yB) for g in Bg]
    ARL_RESULTS[f"K={K} SUP concept"]=eval_detector(f"K={K} SUP concept",cal_sup(),in_sup,chg_sup_concept)
json.dump(ARL_RESULTS,open("results/sbm/sbm_arl2_results.json","w"),indent=2)
print("saved sbm_arl2_results.json")
print("done")

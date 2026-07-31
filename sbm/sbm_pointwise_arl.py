import json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, dgl
from dgl.nn import GraphConv
from scipy.spatial.distance import cdist
PAPER={2:([0.10,0.08],[0.18,0.10],0.05),3:([0.70,0.40,0.10],[0.50,0.30,0.20],0.03),
       5:([0.90,0.70,0.48,0.29,0.10],[0.80,0.60,0.38,0.19,0.10],0.01)}
REORDER={2:[0.05,0.25],3:[0.15,0.45,0.85],5:[0.15,0.35,0.55,0.75,0.95]}
CBASE={2:[0.40,0.10],3:[0.70,0.40,0.10],5:[0.90,0.70,0.48,0.29,0.10]}
N=100; TARGET_ARL=100; MC=30; IN_HORIZON=300; CH=10
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
class GCN(nn.Module):
    def __init__(s,K):super().__init__();s.c1=GraphConv(4,64,allow_zero_in_degree=True);s.c2=GraphConv(64,K,allow_zero_in_degree=True);s.dp=nn.Dropout(0.5)
    def forward(s,g,x):return s.c2(g,s.dp(F.relu(s.c1(g,x))))

def fit_pointwise_ref(refX):
    mu=refX.mean(0);sd=refX.std(0)+1e-9;Rz=(refX-mu)/sd
    d=cdist(Rz,Rz);med=np.median(d)+1e-9;gamma=1/(2*med**2)
    K=np.exp(-gamma*d**2);m=len(Rz)
    krr_term=float((K.sum()-np.trace(K))/(m*(m-1)))
    return {"mu":mu,"sd":sd,"gamma":gamma,"krr_term":krr_term,"Rz":Rz,"m":m}

def pointwise_score(z,ref_stats):
    zz=(z-ref_stats["mu"])/ref_stats["sd"]
    cross=np.exp(-ref_stats["gamma"]*np.sum((ref_stats["Rz"]-zz)**2,axis=1))
    return ref_stats["krr_term"]+1.0-(2.0/ref_stats["m"])*cross.sum()

def score_stream_pointwise(ref_stats,graphs):
    return [pointwise_score(gembed(g),ref_stats) for g in graphs]

def run_cusum(scores,cal,h):
    cal=np.array([max(c,0) for c in cal]);nc=len(cal);S=0.0
    for t,sc in enumerate(scores):
        p=(1+int((cal>=max(sc,0)).sum()))/(nc+1);S=max(0.0,S+np.log(max(-np.log(p),1e-12)))
        if S>h:return t
    return None

def eval_detector(name,cal,score_in,score_change):
    hs=np.arange(0.2,10,0.1);arl={h:[] for h in hs}
    for r in range(MC):
        sc=score_in(r)
        for h in hs:
            fc=run_cusum(sc,cal,h);arl[h].append(fc if fc is not None else len(sc))
    arl_mean={h:np.mean(arl[h]) for h in hs}
    h_star=min(hs,key=lambda h:abs(arl_mean[h]-TARGET_ARL))
    if h_star in (hs[0],hs[-1]): print(f"  WARNING: {name} h*={h_star} hit grid boundary, widen hs",flush=True)
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
    print(f"{name:<30} h*={h_star:.1f}  ARL0(sel)={arl_mean[h_star]:.0f}  "
          f"ARL0(eval)={arl_eval:.0f} 95%CI=({ci[0]:.0f},{ci[1]:.0f}) censored={eval_censored}/{MC}  "
          f"delay={np.mean(delays) if delays else float('nan'):.1f}  det-rate={det/MC:.2f}",flush=True)
    return {"name":name,"h_star":float(h_star),"arl0_selection_batch":float(arl_mean[h_star]),
            "arl0_independent_eval":arl_eval,"arl0_ci95":list(ci),"censored_of_MC":[eval_censored,MC],
            "delay_mean":float(np.mean(delays)) if delays else None,
            "delay_std":float(np.std(delays)) if delays else None,
            "delays_per_run":[int(x) for x in delays],"det_rate":det/MC}

ARL_RESULTS={}
for K in (2,3,5):
    p0,p1,q=PAPER[K];base=np.concatenate([np.full(len(_ix),k) for k,_ix in enumerate(np.array_split(np.arange(N),K))])
    p1r=REORDER[K];cbase=CBASE[K]
    # ---- unsupervised (pointwise): order-kept & reorder ----
    rng0=np.random.default_rng(700+K)
    REF=[sbm(base,p0,q,K,rng0) for _ in range(70)]  # matches sbm_full.py's 70-graph reference
    ref_stats=fit_pointwise_ref(np.array([gembed(g) for g in REF]))
    def cal_pw():
        r=np.random.default_rng(710+K);cp=[sbm(base,p0,q,K,r) for _ in range(64)]
        return score_stream_pointwise(ref_stats,cp)
    def in_pw(r,K=K,q=q,p0=p0,base=base):
        rng=np.random.default_rng(1000+K*100+r);return score_stream_pointwise(ref_stats,[sbm(base,p0,q,K,rng) for _ in range(IN_HORIZON)])
    def chg_ok(r,K=K,q=q,p0=p0,p1=p1,base=base):
        rng=np.random.default_rng(2000+K*100+r);gs=[sbm(base,p0,q,K,rng) for _ in range(CH)]+[sbm(base,p1,q,K,rng) for _ in range(20)];return score_stream_pointwise(ref_stats,gs)
    def chg_re(r,K=K,q=q,p0=p0,p1r=p1r,base=base):
        rng=np.random.default_rng(3000+K*100+r);gs=[sbm(base,p0,q,K,rng) for _ in range(CH)]+[sbm(base,p1r,q,K,rng) for _ in range(20)];return score_stream_pointwise(ref_stats,gs)
    cal=cal_pw()
    ARL_RESULTS[f"K={K} POINTWISE order-kept"]=eval_detector(f"K={K} POINTWISE order-kept",cal,in_pw,chg_ok)
    ARL_RESULTS[f"K={K} POINTWISE reorder"]=eval_detector(f"K={K} POINTWISE reorder",cal,in_pw,chg_re)
    # ---- supervised: concept (already single-snapshot/pointwise; unchanged protocol) ----
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
json.dump(ARL_RESULTS,open("results/sbm/sbm_pointwise_arl_results.json","w"),indent=2)
print("saved sbm_pointwise_arl_results.json")
print("done")

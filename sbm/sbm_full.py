import os, json, argparse
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, dgl
from dgl.nn import GraphConv
from scipy.spatial.distance import cdist
N=100; NG=100
PAPER={2:([0.10,0.08],[0.18,0.10],0.05),
       3:([0.70,0.40,0.10],[0.50,0.30,0.20],0.03),
       5:([0.90,0.70,0.48,0.29,0.10],[0.80,0.60,0.38,0.19,0.10],0.01)}
REORDER={2:[0.05,0.25], 3:[0.15,0.45,0.85], 5:[0.15,0.35,0.55,0.75,0.95]}
# concept/degreeregen need separable communities so degree bins are clean; 2-class raised to a
# paper-used contrast [0.40,0.10] (Table-1's [0.10,0.08] has ~zero contrast). 3/5 keep Table-1 p0.
CBASE={2:[0.40,0.10], 3:[0.70,0.40,0.10], 5:[0.90,0.70,0.48,0.29,0.10]}
def deg(g):
    d=g.in_degrees().float(); dm=d/d.max().clamp(min=1)
    return torch.stack([torch.ones_like(d),dm,dm**2,torch.log1p(d)],1)
def sbm(base,intra,q,K,rng):
    P=np.full((K,K),q,float)
    for k in range(K):P[k,k]=intra[k]
    pr=P[base[:,None],base[None,:]];iu=np.triu_indices(N,1);e=rng.random(len(iu[0]))<pr[iu]
    s=np.concatenate([iu[0][e],iu[1][e]]);d=np.concatenate([iu[1][e],iu[0][e]])
    # degree feature must be the plain graph degree, computed BEFORE
    # add_self_loop (self-loops would add +1 to every node's degree).
    g0=dgl.graph((torch.tensor(s),torch.tensor(d)),num_nodes=N);x=deg(g0)
    g=dgl.add_self_loop(g0);g.ndata["x"]=x;return g
def gen(base,intra,q,K,ng,rng): return [sbm(base,intra,q,K,rng) for _ in range(ng)]
def gembed(g):
    # subtract the self-loop's +1 so this is the plain graph degree.
    d=(g.in_degrees()-1).numpy().astype(float)
    return np.array([d.mean(),d.std(),*np.quantile(d,[.1,.25,.5,.75,.9]),d.max()])
def degree_labels_avg(graphs,K):
    dsum=np.zeros(N)
    for g in graphs: dsum+=(g.in_degrees()-1).numpy()
    order=np.argsort(dsum,kind="stable");lab=np.zeros(N,int)
    for k,idx in enumerate(np.array_split(order,K)):lab[idx]=k
    return lab
def mmd2(X,Y):
    Z=np.concatenate([X,Y]);mu=Z.mean(0);sd=Z.std(0)+1e-9;X=(X-mu)/sd;Y=(Y-mu)/sd
    D=np.concatenate([X,Y]);med=np.median(cdist(D,D))+1e-9;g=1/(2*med**2)
    Kxx=np.exp(-g*cdist(X,X)**2);Kyy=np.exp(-g*cdist(Y,Y)**2);Kxy=np.exp(-g*cdist(X,Y)**2)
    m,n=len(X),len(Y);return float((Kxx.sum()-np.trace(Kxx))/(m*(m-1))+(Kyy.sum()-np.trace(Kyy))/(n*(n-1))-2*Kxy.mean())
class GCN(nn.Module):
    def __init__(s,K):super().__init__();s.c1=GraphConv(4,64,allow_zero_in_degree=True);s.c2=GraphConv(64,K,allow_zero_in_degree=True);s.dp=nn.Dropout(0.5)
    def forward(s,g,x):return s.c2(g,s.dp(F.relu(s.c1(g,x))))
def train(graphs,labels,mask,vg,vl,vmask,K,dev,ep=100,mb=10,mb_rng=None,init=None):
    if mb_rng is None: mb_rng=np.random.default_rng()
    m=GCN(K).to(dev)
    if init is not None: m.load_state_dict(init)
    o=torch.optim.Adam(m.parameters(),lr=1e-2,weight_decay=5e-4);best=-1;bs=None;idx=np.arange(len(graphs))
    for _ in range(ep):
        m.train();mb_rng.shuffle(idx);sel=idx[:mb];o.zero_grad()
        torch.stack([F.cross_entropy(m(graphs[i].to(dev),graphs[i].ndata["x"].to(dev))[mask],labels[i].to(dev)[mask]) for i in sel]).mean().backward();o.step();m.eval()
        with torch.no_grad():
            c=t=0
            for i in range(0,len(vg),5):
                g=vg[i].to(dev);pr=m(g,g.ndata["x"].to(dev))[vmask].argmax(1);c+=(pr==vl[i].to(dev)[vmask]).sum().item();t+=int(vmask.sum())
        a=c/max(t,1)
        if a>best:best=a;bs={k:v.detach().clone() for k,v in m.state_dict().items()}
    if bs:m.load_state_dict(bs)
    return m
def evalacc(m,graphs,labels,mask,dev):
    with torch.no_grad():
        c=t=0
        for i in range(len(graphs)):
            g=graphs[i].to(dev);pr=m(g,g.ndata["x"].to(dev))[mask].argmax(1);c+=(pr==labels[i].to(dev)[mask]).sum().item();t+=int(mask.sum())
    return c/max(t,1)
def cusum_fire(scores_stream, cal, thr=3.0):
    cal=np.array([max(c,0) for c in cal]);nc=len(cal);S=0.0;fired=None
    for t,sc in enumerate(scores_stream):
        p=(1+int((cal>=max(sc,0)).sum()))/(nc+1);S=max(0.0,S+np.log(max(-np.log(p),1e-12)))
        if S>thr and fired is None:fired=t
    return fired,float(S)
def unsup_detect(A,B):
    CHANGE_TIME=10
    ref=np.array([gembed(g) for g in A[:70]]);w=5
    cp=[gembed(g) for g in A[70:90]];cal=[mmd2(np.array(cp[i:i+w]),ref) for i in range(len(cp)-w+1)]
    st=[gembed(g) for g in A[90:100]]+[gembed(g) for g in B[:10]];win=[];scores=[]
    for z in st:
        win.append(z);scores.append(mmd2(np.array(win[-w:]),ref) if len(win)>=w else 0.0)
    fired_sliced,S=cusum_fire(scores[w-1:],cal)
    fired=None if fired_sliced is None else fired_sliced+(w-1)   # true stream-time index
    false_alarm=fired is not None and fired<CHANGE_TIME
    delay=None if (fired is None or false_alarm) else fired-CHANGE_TIME
    post=[s for i,s in enumerate(scores) if i>=CHANGE_TIME]
    return float(np.mean(post)),fired,false_alarm,delay
def sup_detect(model,A,B,ylist,dev,mask):
    def loss_of(g,y):
        with torch.no_grad():
            gg=g.to(dev);return float(F.cross_entropy(model(gg,gg.ndata["x"].to(dev))[mask],y.to(dev)[mask]))
    CHANGE_TIME=10   # stream[0:10]=in-control A, stream[10:20]=post-change B; no window slicing here, so
    yA,yB=ylist      # the raw index returned by cusum_fire is already the true stream-time index.
    cal=[loss_of(g,yA) for g in A[70:90]]                      # in-control losses
    stream=[loss_of(g,yA) for g in A[90:100]]+[loss_of(g,yB) for g in B[:10]]
    fired,S=cusum_fire(stream,cal)
    false_alarm=fired is not None and fired<CHANGE_TIME
    delay=None if (fired is None or false_alarm) else fired-CHANGE_TIME
    return fired,false_alarm,delay,float(np.mean([loss_of(g,yA) for g in A[:10]])),float(np.mean([loss_of(g,yB) for g in B[:10]]))

def run(dev,seeds=5,ep=100):
    idx=np.random.default_rng(12345).permutation(N)
    # 4-way split so the supervised monitor uses a mask (mom) disjoint from both
    # the classifier's train/val mask (trm/vam) and the downstream test mask (tem) -- the
    # frozen model never sees mom's labels/gradients, nor is mom used for checkpoint selection.
    trm=torch.zeros(N,dtype=torch.bool);trm[idx[:30]]=True
    vam=torch.zeros(N,dtype=torch.bool);vam[idx[30:45]]=True
    mom=torch.zeros(N,dtype=torch.bool);mom[idx[45:60]]=True
    tem=torch.zeros(N,dtype=torch.bool);tem[idx[60:]]=True
    R={}
    for K in (2,3,5):
        p0,p1,q=PAPER[K];base=np.concatenate([np.full(len(_ix),k) for k,_ix in enumerate(np.array_split(np.arange(N),K))])  # balanced (differ by <=1), matches paper's 'balanced community' setup
        for ct in ("orderkept","reorder","concept","degreeregen"):
            uM=[];uF=[];uFA=[];uDelay=[];sF=[];sFA=[];sDelay=[];sLA=[];sLB=[];Sc=[];Cu=[];Fo=[]
            for s in range(seeds):
                torch.manual_seed(s)
                rng=np.random.default_rng(700+K*10+s)
                # dedicated local Generator for minibatch shuffling, independent
                # of the graph-generation rng above and of global numpy state.
                mb_rng=np.random.default_rng(9000+K*10+s)
                cp0=CBASE[K] if ct in ("concept","degreeregen") else p0   # concept: separable communities (2-class raised)
                A=gen(base,cp0,q,K,NG,rng);yA=torch.tensor(base)
                if ct=="orderkept": B=gen(base,p1,q,K,NG,rng);yB=torch.tensor(base)
                elif ct=="reorder": B=gen(base,REORDER[K],q,K,NG,rng);yB=torch.tensor(base)
                elif ct=="concept": B=gen(base,cp0,q,K,NG,rng);yB=torch.tensor(degree_labels_avg(B,K))
                else:               # degreeregen: new communities C' = degree bins of A, regenerate edges
                    Cp=degree_labels_avg(A,K);B=gen(Cp,cp0,q,K,NG,rng);yB=torch.tensor(Cp)
                yAl=[yA]*NG; yBl=[yB]*NG
                mm,fu,ufa,udel=unsup_detect(A,B)
                uM.append(mm);uF.append(fu if fu is not None else -1);uFA.append(ufa)
                if udel is not None: uDelay.append(udel)
                # The frozen detection model is trained ONLY on A[:70], matching the
                # unsupervised reference split, so that A[70:90] -- used as the calibration
                # null in sup_detect -- is genuinely held out and never seen during training.
                fixed=train(A[:70],[yA]*70,trm,A[:70],[yA]*70,vam,K,dev,ep,mb_rng=mb_rng)
                fs,sfa,sdel,la,lb=sup_detect(fixed,A,B,(yA,yB),dev,mom)
                sF.append(fs if fs is not None else -1);sFA.append(sfa);sLA.append(la);sLB.append(lb)
                if sdel is not None: sDelay.append(sdel)
                # Fair policy comparison: stale/cumulative/forget all start from the same
                # per-seed initial weights, and each draws its minibatch sequence from a
                # freshly re-seeded (hence identical) generator rather than one shared,
                # sequentially-advancing generator.
                base_state={k:v.clone() for k,v in GCN(K).to(dev).state_dict().items()}
                def mbr(_s=s,_K=K): return np.random.default_rng(9000+_K*10+_s)
                st=train(A,yAl,trm,A,yAl,vam,K,dev,ep,mb_rng=mbr(),init=base_state);Sc.append(evalacc(st,B,yBl,tem,dev))  # stale must validate on A only, never touch B
                cu=train(A+B,yAl+yBl,trm,B,yBl,vam,K,dev,ep,mb_rng=mbr(),init=base_state);Cu.append(evalacc(cu,B,yBl,tem,dev))
                fo=train(B,yBl,trm,B,yBl,vam,K,dev,ep,mb_rng=mbr(),init=base_state);Fo.append(evalacc(fo,B,yBl,tem,dev))
            # store per-seed raw results (not just the mean) alongside std.
            R[f"{K}-{ct}"]={"unsup_mmd":float(np.mean(uM)),"unsup_mmd_std":float(np.std(uM)),
                            "unsup_fire":float(np.median(uF)),
                            "unsup_false_alarm_rate":float(np.mean(uFA)),"unsup_delay":float(np.mean(uDelay)) if uDelay else None,
                            "sup_fire":float(np.median(sF)),"sup_false_alarm_rate":float(np.mean(sFA)),
                            "sup_delay":float(np.mean(sDelay)) if sDelay else None,
                            "sup_lossA":float(np.mean(sLA)),"sup_lossB":float(np.mean(sLB)),
                            "stale":float(np.mean(Sc)),"stale_std":float(np.std(Sc)),
                            "cumulative":float(np.mean(Cu)),"cumulative_std":float(np.std(Cu)),
                            "forget":float(np.mean(Fo)),"forget_std":float(np.std(Fo)),
                            "stale_per_seed":[float(x) for x in Sc],
                            "cumulative_per_seed":[float(x) for x in Cu],
                            "forget_per_seed":[float(x) for x in Fo],
                            "unsup_mmd_per_seed":[float(x) for x in uM],
                            "sup_fire_per_seed":[int(x) for x in sF],
                            "unsup_fire_per_seed":[int(x) for x in uF],
                            "change_time":10}
            print(f"[K={K} {ct}] uMMD={np.mean(uM):.3f} uFire={np.median(uF):.0f}(FA-rate={np.mean(uFA):.2f},delay={np.mean(uDelay) if uDelay else float('nan'):.1f}) "
                  f"sFire={np.median(sF):.0f}(FA-rate={np.mean(sFA):.2f},delay={np.mean(sDelay) if sDelay else float('nan'):.1f}) "
                  f"lossA={np.mean(sLA):.2f}->lossB={np.mean(sLB):.2f}  st/cu/fo={np.mean(Sc):.2f}/{np.mean(Cu):.2f}/{np.mean(Fo):.2f}",flush=True)
    return R
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seeds",type=int,default=5);ap.add_argument("--ep",type=int,default=100);ap.add_argument("--out",default="sbm_full_results.json")
    a=ap.parse_args();R=run(torch.device(a.device),a.seeds,a.ep);json.dump(R,open(a.out,"w"),indent=2);print("saved",a.out)

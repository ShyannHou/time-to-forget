"""Unsupervised conformal + CUSUM change detection on a per-timestep graph stream.

Scores each snapshot by RBF-MMD between its node-feature distribution and an
in-control reference, converts the score to a conformal p-value, and accumulates
evidence with CUSUM. Supports the Elliptic (49 steps) and DBLP (31 steps) streams.
Saves scores, p-values and CUSUM arrays to .npz."""
import os, argparse, pickle
import numpy as np, torch

def rbf_mmd2(X, Y, gamma):
    Kxx=torch.exp(-gamma*torch.cdist(X,X)**2); Kyy=torch.exp(-gamma*torch.cdist(Y,Y)**2)
    Kxy=torch.exp(-gamma*torch.cdist(X,Y)**2)
    m,n=X.shape[0],Y.shape[0]
    sxx=(Kxx.sum()-Kxx.diag().sum())/(m*(m-1)); syy=(Kyy.sum()-Kyy.diag().sum())/(n*(n-1))
    return float(sxx+syy-2*Kxy.mean())

def bootstrap_threshold(cal_pv, horizon, rng, target_alpha=0.10, n_boot=1000, block=2):
    """Calibrate the alarm threshold from the held-out in-control calibration
    p-values themselves, instead of using an illustrative hardcoded constant. Block-
    bootstrap resamples (block size `block`, to retain a little of the in-control serial
    dependence) synthetic in-control monitoring streams of the true monitoring length, runs
    the same Eq.6 CUSUM accumulator on each, and sets the threshold to the (1-target_alpha)
    quantile of the resulting max-CUSUM-reached distribution -- i.e. the threshold a
    genuinely in-control stream of this length would cross with probability ~target_alpha.
    Caveat: with only len(cal_pv) held-out in-control timesteps for real data, this is a
    small-sample calibration, not a substitute for a long independent in-control record."""
    n=len(cal_pv); maxes=[]
    for _ in range(n_boot):
        idx=[]
        while len(idx)<horizon:
            start=rng.integers(0,n); idx+=[ (start+j)%n for j in range(block) ]
        idx=idx[:horizon]; boot_pv=cal_pv[idx]
        s=0.0; m=0.0
        for p in boot_pv:
            s=max(0.0, s+np.log(max(-np.log(max(p,1e-12)),1e-12))); m=max(m,s)
        maxes.append(m)
    return float(np.quantile(maxes,1-target_alpha))

def load_stream(kind, path):
    if kind=="elliptic":
        gs=pickle.load(open(f"{path}/elliptic_graphs.pkl","rb"))
        return [g.ndata["x"].float() for g in gs]
    else:  # dblp: sub_graph_0..T-1
        T=int(pickle.load(open(f"{path}/statistics","rb"))[0])
        feats=[]
        for i in range(T):
            g=pickle.load(open(f"{path}/sub_graph_{i}_by_edges","rb"))
            feats.append(g.ndata["x"].float())
        return feats

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--kind",required=True,choices=["dblp","elliptic"])
    ap.add_argument("--path",required=True)
    ap.add_argument("--incontrol",type=int,default=8)   # first W timesteps = in-control pool
    ap.add_argument("--ref_frac",type=float,default=0.67)  # fraction of in-control pool used as reference; rest = held-out calibration
    ap.add_argument("--sub",type=int,default=400)       # subsample nodes per timestep for MMD
    ap.add_argument("--k",type=float,default=1.0)        # CUSUM drift
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    torch.manual_seed(0); rng=np.random.default_rng(0)
    feats=load_stream(a.kind,a.path); T=len(feats)
    # standardize using ONLY the in-control pool (steps 0..incontrol-1), not the
    # full stream -- using future timesteps' statistics would leak post-change information
    # into the normalization even though it never touches future labels.
    allX=torch.cat(feats[:a.incontrol],0); mu=allX.mean(0); sd=allX.std(0)+1e-6
    feats=[(f-mu)/sd for f in feats]
    def samp(X):
        idx=rng.choice(X.shape[0],min(a.sub,X.shape[0]),replace=False)
        return X[idx]
    # split the in-control pool into a reference block and a DISJOINT held-out
    # calibration block -- calibration timesteps must never contribute to the reference,
    # otherwise calibration scores are in-sample and systematically too small (early false alarms).
    n_ref=max(1,int(round(a.incontrol*a.ref_frac))); n_cal=a.incontrol-n_ref
    if n_cal<1: raise ValueError("incontrol too small for a disjoint ref/calibration split")
    ref_steps=list(range(n_ref)); cal_steps=list(range(n_ref,a.incontrol))
    ref=torch.cat([samp(feats[t]) for t in ref_steps],0)
    if ref.shape[0]>a.sub*2:
        ref=ref[rng.choice(ref.shape[0],a.sub*2,replace=False)]
    # median-heuristic bandwidth on reference
    with torch.no_grad():
        d=torch.pdist(ref); med=float(d.median()); gamma=1.0/(2*med**2+1e-9)
    # per-timestep MMD vs reference (calibration timesteps are OUT of the reference pool)
    scores=np.array([rbf_mmd2(samp(feats[t]), ref, gamma) for t in range(T)])
    # calibration null = held-out (disjoint) in-control timesteps' scores only
    cal=scores[cal_steps]
    pv=np.array([ (1+np.sum(cal>=scores[t]))/(len(cal)+1) for t in range(T) ])
    # for the bootstrap threshold ONLY, calibration points must be scored via
    # leave-one-out against the REST of cal (not the in-sample pv above, which compares a
    # calibration point against a pool that includes itself and is systematically less
    # extreme than a genuine held-out monitoring point's p-value would be).
    pv_cal_loo=np.array([(1+np.sum(np.delete(cal,i)>=cal[i]))/len(cal) for i in range(len(cal))])
    # official monitoring starts only AFTER reference+calibration are both built
    # (t >= monitor_start = a.incontrol); CUSUM is reset to 0 there, so that any noise from
    # the reference/calibration timesteps themselves cannot be miscounted as early monitoring
    # evidence. Paper's literal Eq.6 conformal CUSUM: C_t=max(0,C_{t-1}+log f(p_t)), f(p)=-log(p).
    monitor_start=a.incontrol
    cusum=np.full(T,np.nan); s=0.0
    for t in range(monitor_start,T):
        s=max(0.0, s + np.log(max(-np.log(pv[t]),1e-12))); cusum[t]=s
    # bootstrap-calibrated threshold from the held-out calibration p-values,
    # targeting a ~10% probability of a false alarm over a monitoring stream this long.
    thr_boot=bootstrap_threshold(pv_cal_loo, T-monitor_start, rng, target_alpha=0.10)
    os.makedirs(a.out,exist_ok=True)
    np.savez(f"{a.out}/conformal_{a.kind}.npz", scores=scores, pv=pv, cusum=cusum,
             incontrol=a.incontrol, monitor_start=monitor_start, ref_steps=np.array(ref_steps),
             cal_steps=np.array(cal_steps), thr_illustrative=3.0, thr_bootstrap=thr_boot)
    print(f"=== {a.kind}  T={T}  ref={ref_steps}  cal={cal_steps}  monitor_start={monitor_start} ===")
    print(f"bootstrap-calibrated threshold (target 10% false-alarm prob over horizon): {thr_boot:.2f}  "
          f"(illustrative constant used elsewhere: 3.0)")
    print("t  : MMD    p     CUSUM (nan = before monitoring starts)")
    for t in range(T):
        print(f"{t:>2} : {scores[t]:.3f}  {pv[t]:.3f}  {cusum[t]:.2f}" if t>=monitor_start else f"{t:>2} : {scores[t]:.3f}  {pv[t]:.3f}  --")
    print(f"saved {a.out}/conformal_{a.kind}.npz")

if __name__=="__main__": main()

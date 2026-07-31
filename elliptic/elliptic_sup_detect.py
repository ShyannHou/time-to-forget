import os, argparse, pickle, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, dgl
from dgl.nn import GraphConv
def load(D): return pickle.load(open(f"{D}/elliptic_graphs.pkl","rb"))
def bootstrap_threshold(cal_scores,horizon,rng,target_alpha=0.10,n_boot=1000,block=2):
    cal_scores=np.array(cal_scores);n=len(cal_scores)
    pv_pool=np.array([(1+int((np.delete(cal_scores,i)>=cal_scores[i]).sum()))/(n) for i in range(n)])
    maxes=[]
    for _ in range(n_boot):
        idx=[]
        while len(idx)<horizon:
            start=rng.integers(0,n);idx+=[(start+j)%n for j in range(block)]
        idx=idx[:horizon];boot_pv=pv_pool[idx]
        s=0.0;m=0.0
        for p in boot_pv:
            s=max(0.0,s+np.log(max(-np.log(max(p,1e-12)),1e-12)));m=max(m,s)
        maxes.append(m)
    return float(np.quantile(maxes,1-target_alpha))
class GCN(nn.Module):
    def __init__(s,i,h,c,d=0.5):
        super().__init__();s.c1=GraphConv(i,h,allow_zero_in_degree=True);s.c2=GraphConv(h,c,allow_zero_in_degree=True);s.dp=nn.Dropout(d)
    def forward(s,g,x):return s.c2(g,s.dp(F.relu(s.c1(g,x))))
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train_end",type=int,default=20)   # ts1..train_end used to TRAIN the frozen model
    ap.add_argument("--cal_end",type=int,default=30)      # train_end+1..cal_end = held-out calibration; monitor from cal_end+1
    ap.add_argument("--seed",type=int,default=0)   # expose the seed rather than hardcoding it
    ap.add_argument("--data",default="data/elliptic")   # dir containing elliptic_graphs.pkl
    ap.add_argument("--out",default="results/elliptic") # where the .npz results are written
    a=ap.parse_args()
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gs=load(a.data)
    TRAIN_END=a.train_end; CAL_END=a.cal_end    # ts1-TRAIN_END train; TRAIN_END+1..CAL_END calibration (never trained on); CAL_END+1+ monitoring
    allX=torch.cat([gs[i].ndata["x"] for i in range(TRAIN_END)],0);mu=allX.mean(0);sd=allX.std(0)+1e-6
    P=[]
    for g in gs:
        g=g.to(dev);P.append((g,((g.ndata["x"]-mu.to(dev))/sd.to(dev)),g.ndata["y"].to(dev)))
    IN=list(range(0,TRAIN_END))  # TRAIN the frozen monitor model on ts1-20 ONLY
    wt=torch.tensor([1.,5.],device=dev)
    torch.manual_seed(a.seed);m=GCN(P[0][1].shape[1],128,2).to(dev)
    o=torch.optim.Adam(m.parameters(),lr=5e-3,weight_decay=5e-4)
    for ep in range(150):
        m.train();o.zero_grad()
        L=[F.cross_entropy(m(P[i][0],P[i][1])[P[i][2]>=0],P[i][2][P[i][2]>=0],weight=wt) for i in IN]
        torch.stack(L).mean().backward();o.step()
    m.eval()
    def stats(logit,y):
        msk=y>=0;p=logit[msk].argmax(1);yy=y[msk]
        tp=int(((p==1)&(yy==1)).sum());fp=int(((p==1)&(yy==0)).sum());fn=int(((p==0)&(yy==1)).sum())
        prec=tp/(tp+fp) if tp+fp else 0.;rec=tp/(tp+fn) if tp+fn else 0.
        f1=2*prec*rec/(prec+rec) if prec+rec else 0.
        with torch.no_grad(): loss=float(F.cross_entropy(logit[msk],yy,weight=wt)) if int(msk.sum())>0 else float("nan")
        return f1,prec,rec,loss
    recall=[];precision=[];f1s=[];losses=[]
    with torch.no_grad():
        for i in range(len(P)):
            f1,prec,rec,loss=stats(m(P[i][0],P[i][1]),P[i][2])
            f1s.append(f1);precision.append(prec);recall.append(rec);losses.append(loss)
    recall=np.array(recall);losses=np.array(losses)
    # Calibration null = ts21-30 (CAL_END), which the model NEVER trained on (no
    # train/calibration overlap). Monitoring starts at CAL_END (CUSUM reset there).
    def monitor(score,cal_end=CAL_END):
        cal=score[TRAIN_END:cal_end]
        S=0.0;out=[]
        for t in range(cal_end,len(score)):
            p=(1+int((cal>=score[t]).sum()))/(len(cal)+1);S=max(0.0,S+np.log(max(-np.log(p),1e-12)));out.append(S)
        return np.array(out),cal_end
    cusum_recall,start=monitor(1-recall)
    cusum_loss,_=monitor(losses)
    rng=np.random.default_rng(a.seed)
    # The threshold is calibrated on the score actually used for the reported alarm time
    # (1-recall), not on the loss score: these are two different nonconformity signals.
    thr_boot=bootstrap_threshold((1-recall)[TRAIN_END:CAL_END],len(losses)-CAL_END,rng,target_alpha=0.10)
    os.makedirs(a.out,exist_ok=True)
    tag=f"t{TRAIN_END}c{CAL_END}_seed{a.seed}"
    np.savez(os.path.join(a.out,f"elliptic_sup_detect_{tag}.npz"),
             recall=recall,precision=np.array(precision),f1=np.array(f1s),loss=losses,
             cusum_recall=cusum_recall,cusum_loss=cusum_loss,monitor_start=start,
             train_end=TRAIN_END,cal_end=CAL_END,in_control=np.array(IN),
             thr_illustrative=3.0,thr_bootstrap=thr_boot,seed=a.seed)
    # Keep the seed-0 file at the original, unsuffixed path too, for backward compatibility
    # with plotting scripts that read elliptic_sup_detect_t{TRAIN_END}c{CAL_END}.npz directly.
    if a.seed==0:
        np.savez(os.path.join(a.out,f"elliptic_sup_detect_t{TRAIN_END}c{CAL_END}.npz"),
                 recall=recall,precision=np.array(precision),f1=np.array(f1s),loss=losses,
                 cusum_recall=cusum_recall,cusum_loss=cusum_loss,monitor_start=start,
                 train_end=TRAIN_END,cal_end=CAL_END,in_control=np.array(IN),
                 thr_illustrative=3.0,thr_bootstrap=thr_boot,seed=a.seed)
    print(f"bootstrap-calibrated threshold (target 10% false-alarm prob over horizon): {thr_boot:.2f}  "
          f"(illustrative constant used elsewhere: 3.0)")
    print(f"ts : recall  loss   cusum_recall (train=ts1-{TRAIN_END}, cal=ts{TRAIN_END+1}-{CAL_END}, monitor from ts{CAL_END+1}, seed={a.seed})")
    for t in range(len(recall)):
        c = cusum_recall[t-start] if t>=start else float("nan")
        print(f"{t+1:>2} : {recall[t]:.3f}  {losses[t]:.3f}  {c if t<start else round(c,2)}")
    print(f"saved {os.path.join(a.out, f'elliptic_sup_detect_{tag}.npz')}")
if __name__=="__main__": main()

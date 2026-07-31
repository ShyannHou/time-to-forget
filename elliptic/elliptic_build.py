import pandas as pd, numpy as np, torch, dgl, pickle, os
D=os.environ.get("ELLIPTIC_RAW_DIR", "data/elliptic_raw")
OUT_DIR=os.environ.get("ELLIPTIC_OUT_DIR", "data/elliptic")
os.makedirs(OUT_DIR, exist_ok=True)
feat=pd.read_csv(f"{D}/elliptic_txs_features.csv",header=None)
cls=pd.read_csv(f"{D}/elliptic_txs_classes.csv")
edges=pd.read_csv(f"{D}/elliptic_txs_edgelist.csv")

cmap={"1":1,"2":0,"unknown":-1}         # illicit=1, licit=0, unknown=-1(masked)
txid2y=dict(zip(cls["txId"], cls["class"].map(cmap)))
txid2ts=dict(zip(feat[0], feat[1]))
edges["ts1"]=edges["txId1"].map(txid2ts); edges["ts2"]=edges["txId2"].map(txid2ts)
within=edges[edges["ts1"]==edges["ts2"]]
print(f"edges total {len(edges)}, within-timestep {len(within)} ({len(within)/len(edges):.2%})")

graphs=[]
print("ts   N     E     illicit  licit  unknown")
for t in range(1,50):
    sub=feat[feat[1]==t]; txids=sub[0].values
    id2idx={tx:i for i,tx in enumerate(txids)}
    X=sub.iloc[:,2:].values.astype(np.float32)
    y=np.array([txid2y.get(tx,-1) for tx in txids],dtype=np.int64)
    et=within[within["ts1"]==t]
    src=[id2idx[a] for a in et["txId1"].values]; dst=[id2idx[b] for b in et["txId2"].values]
    g=dgl.add_self_loop(dgl.graph((torch.tensor(src,dtype=torch.long),torch.tensor(dst,dtype=torch.long)),num_nodes=len(txids)))
    g.ndata["x"]=torch.tensor(X); g.ndata["y"]=torch.tensor(y)
    graphs.append(g)
    print(f"{t:>2} {len(txids):>6} {len(src):>6}  {int((y==1).sum()):>6} {int((y==0).sum()):>6}  {int((y==-1).sum()):>6}")
pickle.dump(graphs, open(f"{OUT_DIR}/elliptic_graphs.pkl","wb"))
print(f"saved {OUT_DIR}/elliptic_graphs.pkl ; feature dim =", X.shape[1])

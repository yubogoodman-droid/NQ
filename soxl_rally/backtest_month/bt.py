# 只讀 import live rules.eval_1m (r3)；不下單、不推播
import sys,re,pickle,numpy as np,pandas as pd
sys.path.insert(0,"/workspace/soxl_ma_watch");import rules as LR
raw=pd.read_csv("bingx_1m_raw.csv")
d1=raw.rename(columns={"time":"t"})[["t","open","high","low","close"]].reset_index(drop=True)
for n in (7,14,25,30,200): d1[f"ma{n}"]=d1.close.rolling(n).mean()
def run(allday=False):
    orig=LR.us_session
    if allday: LR.us_session=lambda ms:True
    st=LR.default_state();hits=[]
    def cap(sig,t,title,body):
        if sig=="r3":hits.append((int(t),body))
    for i in range(1,len(d1)): LR.eval_1m(d1,i,st,cap)
    LR.us_session=orig;return hits
res={"us":run(False),"all":run(True)}
pickle.dump(res,open("hits.pkl","wb"));print({k:len(v) for k,v in res.items()})

import sys,requests,time,numpy as np,pandas as pd,matplotlib
matplotlib.use("Agg");import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
sys.path.insert(0,"/workspace/soxl_ma_watch");import rally_rule as RR
fp="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
try: fm.fontManager.addfont(fp); plt.rcParams["font.family"]=fm.FontProperties(fname=fp).get_name()
except Exception: plt.rcParams["font.family"]="Noto Serif CJK TC"
plt.rcParams["axes.unicode_minus"]=False
U="https://open-api.bingx.com/openApi/swap/v3/quote/klines";rows={};end=None
for i in range(10):
    p={"symbol":"NCSKSOXL2USD-USDT","interval":"1m","limit":1440}
    if end:p["endTime"]=end
    d=requests.get(U,params=p,timeout=20).json().get("data") or []
    if not d:break
    for k in d: rows[k["time"]]=k
    end=min(k["time"] for k in d)-1;time.sleep(0.3)
raw=pd.DataFrame(rows.values())
for c in ["open","close","high","low","volume"]: raw[c]=raw[c].astype(float)
raw=raw.sort_values("time").reset_index(drop=True)
now=int(time.time()*1000); raw=raw[raw.time+60000<=now].reset_index(drop=True)  # closed bars only
raw.to_csv("bingx_1m_raw.csv",index=False)
df=RR.features(raw.time,raw.high,raw.low,raw.close); df["open"]=raw.open.values
for n in [14,30,60,120]: df[f"ma{n}"]=df.close.rolling(n).mean()
gaps=(np.diff(df.time)!=60000).sum()
print("bars",len(df),df.dt.iat[0],df.dt.iat[-1],"gaps",gaps)
S=pd.Timestamp("2026-10-24 09:30",tz="Asia/Taipei");E=pd.Timestamp("2026-10-31 09:30",tz="Asia/Taipei")
import rules as LR, re
d1=raw.rename(columns={"time":"t"})[["t","open","high","low","close"]].reset_index(drop=True)
for n in (7,14,25,30,200): d1[f"ma{n}"]=d1.close.rolling(n).mean()
st=LR.default_state(); hits=[]
def cap(sig,t,title,body):
    if sig=="r3": hits.append((t,body))
for i in range(1,len(d1)): LR.eval_1m(d1,i,st,cap)
t2i={int(t):i for i,t in enumerate(df.time)}
sig=[(t2i[t],True,b) for t,b in hits]
cl,lo,hi=df.close.values,df.low.values,df.high.values
def sim(i,tp,slp,maxb=120):
    e=cl[i];sl=slp
    for k in range(i+1,min(i+1+maxb,len(df))):
        if lo[k]<=sl:return sl/e-1,"停損",k
        if hi[k]>=e*(1+tp):return tp,"停利",k
    k=min(i+maxb,len(df)-1);return cl[k]/e-1,"時間出場",k
out=[]
for i,band,body in sig:
    t=df.dt.iat[i]
    if not(S<=t<E) or i<RR.NEED-1:continue
    r=dict(body=body,time=t.strftime("%Y-%m-%d %H:%M"),in_band=band,entry=cl[i],dist_ma200_pct=df.dist_ma200_pct.iat[i])
    sw=df.swinglow.iat[i]*0.999
    ret,why,k=sim(i,0.03,sw);r.update(sl=round(sw,3),tp=round(cl[i]*1.03,3),exit_reason=why,exit_time=df.dt.iat[k].strftime("%m-%d %H:%M"),exit_px=round(cl[i]*(1+ret),3),hold_min=k-i,gross_pct=round(ret*100,3),net_pct=round(ret*100-0.1,3))
    L2=float(re.search(r'低2 ([\d.]+)',body).group(1));sl2=max(L2,cl[i]*0.99);r['L2']=L2;ret2,why2,k2=sim(i,0.01,sl2);r.update(sens_net_pct=round(ret2*100-0.1,3),sens_exit=why2,sens_hold=k2-i,complete=(i+120<len(df)))
    r["_i"]=i;r["_k"]=k;out.append(r)
T=pd.DataFrame(out);print(T.drop(columns=["_i","_k"]).to_string())
for _,r in T.iterrows():
    i,k=r._i,r._k;w=df.iloc[max(0,i-150):min(len(df),k+30)];x=w.dt.dt.tz_localize(None)
    fig,ax=plt.subplots(figsize=(16,8))
    for xi,o,h,l,c in zip(x,w.open,w.high,w.low,w.close):
        col="tab:green" if c>=o else "tab:red";ax.vlines(xi,l,h,color=col,lw=.8);ax.vlines(xi,min(o,c),max(o,c),color=col,lw=3)
    for n,cc in zip([7,14,25,30,60,120,200],["gold","magenta","deepskyblue","teal","orange","brown","purple"]):ax.plot(x,w[f"ma{n}"],color=cc,lw=1.3,label=f"MA{n}")
    ti=df.dt.iat[i].tz_localize(None);tk=df.dt.iat[k].tz_localize(None)
    ax.axvline(ti,color="blue",ls=":",lw=1);ax.scatter(ti,r.entry,marker="^",s=260,color="blue",zorder=6,label=f"通知/進場 {r.entry:.2f}")
    ax.axhline(r.sl,color="red",ls="--",label=f"停損 {r.sl:.2f}");ax.axhline(r.tp,color="green",ls="--",label=f"停利+3% {r.tp:.2f}")
    ax.scatter(tk,r.exit_px,marker="X",s=260,color="black",zorder=6,label=f"出場({r.exit_reason}) {r.exit_px:.2f}")
    ax.set_title(f"SOXL 一分K 起漲點 {r.time} 台北時間  淨{r.net_pct:+.2f}%  ",fontsize=17)
    ax.tick_params(labelsize=13);ax.legend(loc="upper left",fontsize=12,ncol=2);ax.grid(alpha=.3);fig.tight_layout()
    fig.savefig(f"charts/{r.time.replace(' ','_').replace(':','')}.png",dpi=200);plt.close(fig)
T.drop(columns=["_i","_k"]).to_csv("signals.csv",index=False)

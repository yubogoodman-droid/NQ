import sys,re,pickle,numpy as np,pandas as pd,matplotlib
matplotlib.use("Agg");import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.backends.backend_pdf import PdfPages
sys.path.insert(0,"/workspace/soxl_ma_watch");import rally_rule as RR
fp="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc";fm.fontManager.addfont(fp)
plt.rcParams["font.family"]=fm.FontProperties(fname=fp).get_name();plt.rcParams["axes.unicode_minus"]=False
raw=pd.read_csv("bingx_1m_raw.csv")
df=RR.features(raw.time,raw.high,raw.low,raw.close);df["open"]=raw.open.values
for n in [7,14,25,30,60,120,200]: df[f"ma{n}"]=df.close.rolling(n).mean()
S=pd.Timestamp("2026-10-01 09:30",tz="Asia/Taipei");E=pd.Timestamp("2026-10-31 09:30",tz="Asia/Taipei")
H=pickle.load(open("hits.pkl","rb"));t2i={int(t):i for i,t in enumerate(df.time)}
cl,lo,hi=df.close.values,df.low.values,df.high.values
# live log
live={}
for ln in open("/workspace/soxl_ma_watch/watch.log",encoding="utf-8"):
    m=re.search(r"^(\S+ \S+),\d+ (ALERT|STALE) r3 bar=(\d\d-\d\d \d\d:\d\d)",ln)
    if m: live["2026-"+m.group(3)]=(m.group(2),m.group(1))
LOGSTART=pd.Timestamp("2026-10-21 14:07",tz="Asia/Taipei")
def sim(i,tp,sl,maxb=120):
    e=cl[i]
    for k in range(i+1,min(i+1+maxb,len(df))):
        if lo[k]<=sl:return sl/e-1,"停損",k
        if hi[k]>=e*(1+tp):return tp,"停利",k
    k=min(i+maxb,len(df)-1);return cl[k]/e-1,"時間出場",k
def build(hits):
    out=[]
    for t,body in hits:
        i=t2i[t];tt=df.dt.iat[i]
        if not(S<=tt<E):continue
        r=dict(time=tt.strftime("%Y-%m-%d %H:%M"),body=body,entry=cl[i])
        sw=df.swinglow.iat[i]*0.999;ret,why,k=sim(i,0.03,sw)
        r.update(sl=round(sw,3),tp=round(cl[i]*1.03,3),exit_reason=why,exit_time=df.dt.iat[k].strftime("%m-%d %H:%M"),exit_px=round(cl[i]*(1+ret),3),hold_min=k-i,gross_pct=round(ret*100,3),net_pct=round(ret*100-0.1,3))
        L2=float(re.search(r'低2 ([\d.]+)',body).group(1));sl2=max(L2,cl[i]*0.99);ret2,why2,k2=sim(i,0.01,sl2)
        r.update(L2=L2,sens_net_pct=round(ret2*100-0.1,3),sens_exit=why2,sens_hold=k2-i)
        st={"ALERT":"已推播","STALE":"STALE未推播"}.get(live.get(r["time"],("",))[0],"live無紀錄(10/21前)" if tt<LOGSTART else "log無此筆")
        r["live_status"]=st;r["us_session"]=bool(__import__("rules").us_session(t)) if "/workspace/soxl_ma_watch" in sys.path else None
        r["session_day"]=(tt-pd.Timedelta(hours=6)).strftime("%m-%d");r["week"]=(tt-pd.Timedelta(hours=6)).strftime("%G-W%V")
        r["_i"]=i;r["_k"]=k;out.append(r)
    return pd.DataFrame(out)
def stats(T,col="net_pct",lab=""):
    if len(T)==0:return f"{lab}: 0 筆"
    w=(T[col]>0).sum();return f"{lab}: n={len(T)}，勝 {w}/敗 {len(T)-w}，勝率 {w/len(T)*100:.0f}%，平均 {T[col].mean():+.2f}%，總 {T[col].sum():+.2f}%"
T=build(H["us"]);A=build(H["all"])
T.drop(columns=["_i","_k"]).to_csv("signals.csv",index=False);A.drop(columns=["_i","_k"]).to_csv("signals_allday.csv",index=False)
L=[]
L.append(stats(T,lab="主設定 美股時段(live)"));L.append(stats(T,"sens_net_pct","敏感度"))
L.append(f"最大賺 {T.net_pct.max():+.2f}% 最大賠 {T.net_pct.min():+.2f}% 平均持有 {T.hold_min.mean():.1f}m 出場 {T.exit_reason.value_counts().to_dict()} 敏感度出場 {T.sens_exit.value_counts().to_dict()}")
for s,g in T.groupby("live_status"):L.append(stats(g,lab=f"[{s}]"))
L.append(stats(A,lab="全部時段(去掉us_session)"));L.append(stats(A[~A.time.isin(T.time)],lab="其中非美股時段"));L.append(stats(A,"sens_net_pct","全部時段 敏感度"))
L.append("依週:");[L.append(stats(g,lab="  "+w)) for w,g in T.groupby("week")]
L.append("依日(美股交易夜,以台北開盤日):");[L.append(stats(g,lab="  "+d)) for d,g in T.groupby("session_day")]
L.append("一週區間(10/24-10/31):"+stats(T[T.time>="2026-10-24 09:30"],lab=""))
open("stats.txt","w").write("\n".join(L));print("\n".join(L))
print(T[["time","entry","sl","tp","exit_reason","exit_time","hold_min","net_pct","sens_net_pct","sens_exit","live_status"]].to_string())
def chart(r,path,small=False):
    i,k=r._i,r._k;w=df.iloc[max(0,i-150):min(len(df),k+30)];x=w.dt.dt.tz_localize(None)
    fig,ax=plt.subplots(figsize=(16,8))
    for xi,o,h,l,c in zip(x,w.open,w.high,w.low,w.close):
        col="tab:green" if c>=o else "tab:red";ax.vlines(xi,l,h,color=col,lw=.8);ax.vlines(xi,min(o,c),max(o,c),color=col,lw=3)
    for n,cc in zip([7,14,25,30,60,120,200],["gold","magenta","deepskyblue","teal","orange","brown","purple"]):ax.plot(x,w[f"ma{n}"],color=cc,lw=1.3,label=f"MA{n}")
    ti=df.dt.iat[i].tz_localize(None);tk=df.dt.iat[k].tz_localize(None)
    ax.axvline(ti,color="blue",ls=":",lw=1);ax.scatter(ti,r.entry,marker="^",s=260,color="blue",zorder=6,label=f"通知/進場 {r.entry:.2f}")
    ax.axhline(r.sl,color="red",ls="--",label=f"停損 {r.sl:.2f}");ax.axhline(r.tp,color="green",ls="--",label=f"停利+3% {r.tp:.2f}")
    ax.scatter(tk,r.exit_px,marker="X",s=260,color="black",zorder=6,label=f"出場({r.exit_reason}) {r.exit_px:.2f}")
    ax.set_title(f"SOXL 一分K 起漲點 {r.time} 台北時間  淨{r.net_pct:+.2f}%  {r.live_status}",fontsize=17)
    ax.tick_params(labelsize=13);ax.legend(loc="upper left",fontsize=12,ncol=2);ax.grid(alpha=.3);fig.tight_layout()
    fig.savefig(path,dpi=200);plt.close(fig)
import os
if len(T)>30: sel=pd.concat([T.nlargest(8,"net_pct"),T.nsmallest(8,"net_pct")])
else: sel=T
files=[]
for _,r in sel.iterrows():
    p=f"charts/{r.time.replace(' ','_').replace(':','')}.png";chart(r,p);files.append(p)
# thumbnail overview
n=len(T);cols=5;rows=-(-n//cols);fig,axs=plt.subplots(rows,cols,figsize=(cols*5,rows*3.4))
for a,(_,r) in zip(axs.flat,T.iterrows()):
    i,k=r._i,r._k;w=df.iloc[max(0,i-120):min(len(df),k+20)]
    a.plot(range(len(w)),w.close.values,color="k",lw=.8);a.plot(range(len(w)),w.ma200.values,color="purple",lw=.8)
    a.axvline(i-w.index[0],color="b",ls=":");a.axhline(r.sl,color="r",ls="--",lw=.8);a.axhline(r.tp,color="g",ls="--",lw=.8)
    a.set_title(f"{r.time[5:]} {r.net_pct:+.2f}% {r.exit_reason}",fontsize=11,color="green" if r.net_pct>0 else "red");a.set_xticks([])
for a in list(axs.flat)[n:]:a.axis("off")
fig.suptitle("SOXL 一分K 起漲點 30天全部訊號縮圖（黑=收盤，紫=MA200）",fontsize=18);fig.tight_layout();fig.savefig("charts/overview.png",dpi=200);plt.close(fig)
with PdfPages("all_charts.pdf") as pdf:
    for p in ["charts/overview.png"]+files:
        im=plt.imread(p);fig=plt.figure(figsize=(16,9));plt.imshow(im);plt.axis("off");pdf.savefig(fig,dpi=200);plt.close(fig)
print("charts",len(files))

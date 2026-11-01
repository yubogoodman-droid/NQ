import requests,time,pandas as pd
U="https://open-api.bingx.com/openApi/swap/v3/quote/klines";rows={};end=None
STOP=int(pd.Timestamp("2026-09-29 00:00",tz="Asia/Taipei").timestamp()*1000)
for i in range(60):
    p={"symbol":"NCSKSOXL2USD-USDT","interval":"1m","limit":1440}
    if end:p["endTime"]=end
    d=requests.get(U,params=p,timeout=20).json().get("data") or []
    if not d:print("empty",i);break
    n0=len(rows)
    for k in d: rows[k["time"]]=k
    mn=min(k["time"] for k in d)
    if len(rows)==n0 or mn<STOP:break
    end=mn-1;time.sleep(0.3)
raw=pd.DataFrame(rows.values())
for c in ["open","close","high","low","volume"]: raw[c]=raw[c].astype(float)
raw=raw.sort_values("time").reset_index(drop=True)
now=int(time.time()*1000);raw=raw[raw.time+60000<=now].reset_index(drop=True)
raw.to_csv("bingx_1m_raw.csv",index=False)
print(len(raw),pd.to_datetime(raw.time.iat[0],unit="ms",utc=True).tz_convert("Asia/Taipei"),pd.to_datetime(raw.time.iat[-1],unit="ms",utc=True).tz_convert("Asia/Taipei"))
import numpy as np;g=np.diff(raw.time.values);print("gaps",(g!=60000).sum(),g[g!=60000][:10]/60000)

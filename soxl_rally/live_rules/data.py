"""BingX public swap klines (read-only; no trading)."""
import time, requests, pandas as pd
SYM = "NCSKSOXL2USD-USDT"
URL = "https://open-api.bingx.com/openApi/swap/v3/quote/klines"

def fetch_1m(minutes=2880):
    end = int(time.time() * 1000)
    start = end - minutes * 60000
    rows, cur_end = {}, end
    while cur_end > start:
        r = requests.get(URL, params=dict(symbol=SYM, interval="1m", limit=1440,
                         startTime=max(start, cur_end - 1440 * 60000), endTime=cur_end), timeout=15)
        d = r.json()
        if d.get("code") != 0:
            raise RuntimeError(f"bingx {d}")
        data = d["data"]
        if not data:
            break
        for k in data:
            rows[int(k["time"])] = k
        cur_end = min(int(k["time"]) for k in data) - 1
    df = pd.DataFrame(rows.values())
    df["t"] = df["time"].astype("int64")
    for c in ("open", "high", "low", "close"):
        df[c] = df[c].astype(float)
    return df.sort_values("t").drop_duplicates("t").reset_index(drop=True)[["t", "open", "high", "low", "close"]]

def closed_1m(df, now_ms):
    return df[df.t + 60000 <= now_ms].reset_index(drop=True)

def resample_5m(df1, now_ms):
    g = df1.assign(b=df1.t // 300000 * 300000).groupby("b")
    df5 = pd.DataFrame({"open": g.open.first(), "high": g.high.max(), "low": g.low.min(),
                        "close": g.close.last(), "n": g.size()}).reset_index().rename(columns={"b": "t"})
    return df5[df5.t + 300000 <= now_ms].reset_index(drop=True)

def add_ma(df, ns):
    for n in ns:
        df[f"ma{n}"] = df.close.rolling(n).mean()
    return df

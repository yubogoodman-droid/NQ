"""Fetch BingX NCSKSOXL2USD-USDT 1m klines forward to latest closed bar (read-only public quote API).

Reuses ../backtest_3m/bingx_1m_3m_raw.csv (2026-07-28 18:00 .. 2026-10-31 09:52 UTC+8)
so earlier months stay bit-for-bit identical, and fetches forward through latest closed 1m bar.
Also inspects gaps and abnormal 1m price jumps (e.g. ETF splits / contract resets).
"""
from pathlib import Path
import time
import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
M3_CSV = HERE.parent / "backtest_3m" / "bingx_1m_3m_raw.csv"
OUT_CSV = HERE / "bingx_1m_4m_raw.csv"
URL = "https://open-api.bingx.com/openApi/swap/v3/quote/klines"

src = OUT_CSV if OUT_CSV.exists() else M3_CSV
base = pd.read_csv(src)
rows = {int(r.time): dict(open=float(r.open), close=float(r.close), high=float(r.high),
                          low=float(r.low), volume=float(r.volume), time=int(r.time))
        for _, r in base.iterrows()}

last_have = int(base.time.max())
now_ms = int(time.time() * 1000)
# Only keep closed 1m bars
end_target = (now_ms // 60000 - 1) * 60000
print(f"Seeded {len(rows)} bars from {src.name} (up to "
      f"{pd.to_datetime(last_have, unit='ms', utc=True).tz_convert('Asia/Taipei')}); fetching forward to "
      f"{pd.to_datetime(end_target, unit='ms', utc=True).tz_convert('Asia/Taipei')} ...")

sess = requests.Session()
cur_end = end_target
for step in range(80):
    if cur_end <= last_have:
        break
    start_ms = max(last_have, cur_end - 1439 * 60000)
    data = None
    for retry in range(6):
        resp = sess.get(URL, params=dict(symbol="NCSKSOXL2USD-USDT", interval="1m", limit=1440,
                                         startTime=start_ms, endTime=cur_end), timeout=20).json()
        if resp.get("code") == 0 and resp.get("data"):
            data = resp["data"]
            break
        time.sleep(1.5 * (retry + 1))
    if not data:
        print("Stopped at step", step, resp)
        break
    for k in data:
        t = int(k["time"])
        if t <= end_target and t not in rows:
            rows[t] = dict(open=float(k["open"]), close=float(k["close"]), high=float(k["high"]),
                           low=float(k["low"]), volume=float(k["volume"]), time=t)
    mn = min(int(k["time"]) for k in data)
    if mn >= cur_end:
        break
    cur_end = mn - 1
    if step % 5 == 0:
        print(f"  step {step:2d}: total={len(rows):6d} cur_earliest={pd.to_datetime(mn, unit='ms', utc=True).tz_convert('Asia/Taipei')}")
    time.sleep(0.35)

raw = pd.DataFrame(list(rows.values())).sort_values("time").reset_index(drop=True)
raw = raw[raw.time + 60000 <= now_ms].reset_index(drop=True)
raw = raw[["open", "close", "high", "low", "volume", "time"]]
raw.to_csv(OUT_CSV, index=False)

g = np.diff(raw.time.values)
gaps = np.where(g != 60000)[0]
print(f"Saved {len(raw)} bars: {pd.to_datetime(raw.time.iat[0], unit='ms', utc=True).tz_convert('Asia/Taipei')} -> "
      f"{pd.to_datetime(raw.time.iat[-1], unit='ms', utc=True).tz_convert('Asia/Taipei')} | gaps={len(gaps)}")
if len(gaps):
    for gi in gaps[:10]:
        t0 = pd.to_datetime(raw.time.iat[gi], unit="ms", utc=True).tz_convert("Asia/Taipei")
        t1 = pd.to_datetime(raw.time.iat[gi + 1], unit="ms", utc=True).tz_convert("Asia/Taipei")
        print(f"  gap: {t0} -> {t1} ({(raw.time.iat[gi+1]-raw.time.iat[gi])/60000:.0f}m)")

# Check large 1m jumps (> 10%) to detect splits or anomalies
ret1m = raw.close.pct_change().abs()
jumps = raw[ret1m > 0.10]
print(f"1m jumps > 10%: {len(jumps)}")
for idx in jumps.index[:10]:
    dt = pd.to_datetime(raw.time.iat[idx], unit="ms", utc=True).tz_convert("Asia/Taipei")
    print(f"  jump at {dt}: {raw.close.iat[idx-1]:.2f} -> {raw.close.iat[idx]:.2f} ({(raw.close.iat[idx]/raw.close.iat[idx-1]-1)*100:+.1f}%)")

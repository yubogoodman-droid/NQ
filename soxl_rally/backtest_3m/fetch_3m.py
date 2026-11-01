"""Fetch 3 months of BingX NCSKSOXL2USD-USDT 1m klines (read-only public quote API).

Reuses ../backtest_month/bingx_1m_raw.csv (2026-09-28 10:25 .. 2026-10-31 09:52 UTC+8)
so that the October window stays bit-for-bit identical to backtest_month, and backfills
earlier bars back to 2026-07-29 00:00 UTC+8 (warmup before 2026-07-31 09:30 UTC+8).
"""
from pathlib import Path
import time
import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
M1_CSV = HERE.parent / "backtest_month" / "bingx_1m_raw.csv"
OUT_CSV = HERE / "bingx_1m_3m_raw.csv"
URL = "https://open-api.bingx.com/openApi/swap/v3/quote/klines"
STOP_MS = int(pd.Timestamp("2026-07-29 00:00", tz="Asia/Taipei").timestamp() * 1000)

src = OUT_CSV if OUT_CSV.exists() else M1_CSV
m1 = pd.read_csv(src)
rows = {int(r.time): dict(open=float(r.open), close=float(r.close), high=float(r.high),
                          low=float(r.low), volume=float(r.volume), time=int(r.time))
        for _, r in m1.iterrows()}
cur_end = int(m1.time.min()) - 1
print(f"Seeded {len(rows)} bars from {src.name}; backfilling before "
      f"{pd.to_datetime(cur_end, unit='ms', utc=True).tz_convert('Asia/Taipei')} ...")

sess = requests.Session()
for step in range(120):
    if cur_end < STOP_MS:
        break
    start_ms = max(STOP_MS - 360 * 60000, cur_end - 1439 * 60000)
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
        if t not in rows:
            rows[t] = dict(open=float(k["open"]), close=float(k["close"]), high=float(k["high"]),
                           low=float(k["low"]), volume=float(k["volume"]), time=t)
    mn = min(int(k["time"]) for k in data)
    if mn >= cur_end:
        break
    cur_end = mn - 1
    if step % 10 == 0:
        print(f"  step {step:2d}: total={len(rows):6d} earliest={pd.to_datetime(mn, unit='ms', utc=True).tz_convert('Asia/Taipei')}")
    time.sleep(0.35)

raw = pd.DataFrame(list(rows.values())).sort_values("time").reset_index(drop=True)
raw = raw[["open", "close", "high", "low", "volume", "time"]]
raw.to_csv(OUT_CSV, index=False)
g = np.diff(raw.time.values)
print(f"Saved {len(raw)} bars: {pd.to_datetime(raw.time.iat[0], unit='ms', utc=True).tz_convert('Asia/Taipei')} -> "
      f"{pd.to_datetime(raw.time.iat[-1], unit='ms', utc=True).tz_convert('Asia/Taipei')} | gaps={int((g != 60000).sum())}")

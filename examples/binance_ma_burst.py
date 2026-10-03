#!/usr/bin/env python3
"""幣安 15 分 K：爆量，一根 K 從六條均線下方收到上方。

條件就這兩句：

  • 開盤 ≤ MA7、MA14、MA25、MA99、MA120、MA200，收盤高於這六條。
  • 成交量 ≥ 前 20 根均量的 5 倍。

六條若擠在一起，一根普通的陽線就跨得過去，所以要爆量才算。
已經站上全部均線之後的大陽是延伸，不是突破。

    python3 examples/binance_ma_burst.py --symbol AINUSDT
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import numpy as np
import requests

MA_PERIODS = (7, 14, 25, 99, 120, 200)
VOL_LOOKBACK = 20
VOL_MULT = 5.0
INTERVAL_MS = 15 * 60_000

TZ = timezone(timedelta(hours=8))
BASE = "https://www.binance.com"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0", "Accept": "application/json"})


def sma(a: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(a), np.nan)
    if len(a) >= n:
        out[n - 1 :] = np.convolve(a, np.ones(n) / n, mode="valid")
    return out


def burst_at(
    o: np.ndarray,
    c: np.ndarray,
    v: np.ndarray,
    ma: dict[int, np.ndarray],
    i: int,
    vol_mult: float = VOL_MULT,
) -> dict | None:
    """一根 15 分 K 開在六條均線下、收在六條上，且量是前 20 根的 vol_mult 倍。"""
    need = max(MA_PERIODS) - 1
    if i < max(need, VOL_LOOKBACK) or i >= len(c):
        return None
    vals = np.array([ma[n][i] for n in MA_PERIODS], dtype=float)
    if np.isnan(vals).any():
        return None
    if not (c[i] > o[i] and np.all(o[i] <= vals) and np.all(c[i] > vals)):
        return None
    base = float(v[i - VOL_LOOKBACK : i].mean())
    if base <= 0:
        return None
    vr = float(v[i] / base)
    if vr < vol_mult:
        return None
    return {
        "i": i,
        "open": float(o[i]),
        "close": float(c[i]),
        "body": float(c[i] / o[i] - 1.0),
        "vol_ratio": vr,
        "mas": {n: float(ma[n][i]) for n in MA_PERIODS},
    }


def find_bursts(
    o: np.ndarray,
    c: np.ndarray,
    v: np.ndarray,
    vol_mult: float = VOL_MULT,
) -> list[dict]:
    ma = {n: sma(c, n) for n in MA_PERIODS}
    hits = []
    for i in range(len(c)):
        hit = burst_at(o, c, v, ma, i, vol_mult=vol_mult)
        if hit:
            hits.append(hit)
    return hits


def get_json(path: str, params=None, retries: int = 5):
    last = None
    for i in range(retries):
        try:
            r = SESSION.get(BASE + path, params=params, timeout=20)
            if r.status_code == 429:
                time.sleep(1.3 * (i + 1))
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            time.sleep(0.4 * (i + 1))
    raise last


def fetch_klines(sym: str, limit: int = 1500) -> dict | None:
    raw = get_json("/fapi/v1/klines", params={"symbol": sym, "interval": "15m", "limit": limit})
    if not raw or len(raw) < max(MA_PERIODS) + VOL_LOOKBACK:
        return None
    now_ms = int(time.time() * 1000)
    if int(raw[-1][0]) + INTERVAL_MS > now_ms:
        raw = raw[:-1]
    if len(raw) < max(MA_PERIODS) + VOL_LOOKBACK:
        return None
    return {
        "t": np.array([int(x[0]) for x in raw], np.int64),
        "o": np.array([float(x[1]) for x in raw]),
        "h": np.array([float(x[2]) for x in raw]),
        "l": np.array([float(x[3]) for x in raw]),
        "c": np.array([float(x[4]) for x in raw]),
        "v": np.array([float(x[5]) for x in raw]),
    }


def fmt_hit(sym: str, d: dict, hit: dict) -> str:
    ts = datetime.fromtimestamp(int(d["t"][hit["i"]]) / 1000, TZ).strftime("%Y-%m-%d %H:%M")
    mas = "  ".join(f"MA{n} {hit['mas'][n]:.6g}" for n in MA_PERIODS)
    return (
        f"{sym}  15m  {ts}\n"
        f"開 {hit['open']:.6g} → 收 {hit['close']:.6g}  ({hit['body'] * 100:+.2f}%)\n"
        f"量是前 {VOL_LOOKBACK} 根的 {hit['vol_ratio']:.1f} 倍\n"
        f"{mas}"
    )


def check_symbol(sym: str, limit: int = 1500) -> list[str]:
    d = fetch_klines(sym, limit=limit)
    if d is None:
        return []
    lines = []
    for hit in find_bursts(d["o"], d["c"], d["v"]):
        lines.append(fmt_hit(sym, d, hit))
    return lines


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="15 分 K 爆量，一根穿 7/14/25/99/120/200")
    p.add_argument("--symbol", default="AINUSDT", help="永續代號，例如 AINUSDT")
    p.add_argument("--limit", type=int, default=1500, help="往回看幾根 15 分 K，最多 1500")
    args = p.parse_args()
    sym = args.symbol.upper()
    lines = check_symbol(sym, limit=args.limit)
    if not lines:
        print(f"{sym} 這段 15 分 K 沒有「爆量一根穿六條均線」")
        return 0
    print(f"{sym} 抓到 {len(lines)} 根\n")
    print("\n\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

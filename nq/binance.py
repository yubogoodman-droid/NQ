"""幣安 U 本位永續：標的清單與 K 線。"""

from __future__ import annotations

import time

import numpy as np
import requests

BASE = "https://www.binance.com"
SESSION = requests.Session()
SESSION.headers.update(
    {"User-Agent": "Mozilla/5.0", "Clienttype": "web", "Accept": "application/json"}
)

BARS_PER_DAY = {"15m": 96, "5m": 288, "1m": 1440, "1h": 24, "4h": 6, "1d": 1}
INTERVAL_MS = {
    "15m": 15 * 60_000,
    "5m": 5 * 60_000,
    "1m": 60_000,
    "1h": 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
}


def get_json(path: str, params=None, retries: int = 6):
    last = None
    for i in range(retries):
        try:
            r = SESSION.get(BASE + path, params=params, timeout=25)
            if r.status_code == 429:
                time.sleep(1.4 * (i + 1))
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            time.sleep(0.45 * (i + 1))
    raise last


def universe(*, top_n: int = 50) -> list[tuple[str, float]]:
    """USDT 永續，依 24h 成交額（quoteVolume）由高到低取前 `top_n`。"""
    info = get_json("/fapi/v1/exchangeInfo")
    tickers = {t["symbol"]: t for t in get_json("/fapi/v1/ticker/24hr")}
    ranked: list[tuple[float, str]] = []
    for s in info["symbols"]:
        if s.get("quoteAsset") != "USDT":
            continue
        if s.get("status") != "TRADING":
            continue
        if s.get("contractType") != "PERPETUAL":
            continue
        if s.get("underlyingType") == "INDEX":
            continue
        sym = s["symbol"]
        qv = float((tickers.get(sym) or {}).get("quoteVolume") or 0)
        ranked.append((qv, sym))
    ranked.sort(reverse=True)
    return [(sym, qv) for qv, sym in ranked[:top_n]]


def _to_ohlcv(rows: list) -> dict:
    return {
        "t": np.array([int(x[0]) for x in rows], np.int64),
        "o": np.array([float(x[1]) for x in rows]),
        "h": np.array([float(x[2]) for x in rows]),
        "l": np.array([float(x[3]) for x in rows]),
        "c": np.array([float(x[4]) for x in rows]),
        "v": np.array([float(x[5]) for x in rows]),
    }


def fetch_klines(
    sym: str,
    *,
    interval: str = "1m",
    limit: int | None = None,
    days: int | None = None,
    extra_bars: int = 8,
) -> dict | None:
    """抓已收盤 K 線。`days` 會往回補到足夠根數；`limit` 則單次或分頁到該根數。"""
    if days is not None:
        need = days * BARS_PER_DAY[interval] + extra_bars
    else:
        need = limit or 500
    chunks: list[list] = []
    end_time = None
    while sum(len(c) for c in chunks) < need:
        params = {"symbol": sym, "interval": interval, "limit": min(1500, need)}
        if end_time is not None:
            params["endTime"] = end_time
        raw = get_json("/fapi/v1/klines", params)
        if not raw:
            break
        chunks.append(raw)
        if len(raw) < params["limit"]:
            break
        end_time = int(raw[0][0]) - 1
    rows = []
    seen: set[int] = set()
    for chunk in reversed(chunks):
        for x in chunk:
            t = int(x[0])
            if t in seen:
                continue
            seen.add(t)
            rows.append(x)
    if not rows:
        return None
    now_ms = int(time.time() * 1000)
    if int(rows[-1][0]) + INTERVAL_MS[interval] > now_ms:
        rows = rows[:-1]
    rows = rows[-need:]
    if not rows:
        return None
    return _to_ohlcv(rows)

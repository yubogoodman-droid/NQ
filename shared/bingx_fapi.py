#!/usr/bin/env python3
"""BingX Swap V2 → Binance fapi shim for 1H MA25 破底翻 (stock perps / TradFi).

Translates Binance fapi endpoints used by `binance_1h_ma25_reclaim.py`:
  - `/fapi/v1/exchangeInfo` → BingX `/openApi/swap/v2/quote/contracts` (NCSK* stock contracts)
  - `/fapi/v1/ticker/24hr`  → BingX `/openApi/swap/v2/quote/ticker`
  - `/fapi/v1/klines`       → BingX `/openApi/swap/v2/quote/klines` (multi-page up to 1500 bars)

Maps BingX internal symbols (e.g. `NCSKTSLA2USD-USDT`, `displayName="TSLA-USDT"`)
to clean Binance-style display symbols (`TSLAUSDT`) and vice versa.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List

import requests

BASE = "https://open-api.bingx.com"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0", "Accept": "application/json"})

_SYM_TO_BINGX: Dict[str, str] = {}
_BINGX_TO_SYM: Dict[str, str] = {}
_CONTRACTS_CACHE: List[dict] = []


def _ensure_contracts() -> List[dict]:
    global _CONTRACTS_CACHE
    if _CONTRACTS_CACHE:
        return _CONTRACTS_CACHE
    r = SESSION.get(f"{BASE}/openApi/swap/v2/quote/contracts", timeout=25)
    r.raise_for_status()
    data = r.json().get("data") or []
    _CONTRACTS_CACHE = data
    for c in data:
        bsym = c.get("symbol") or ""
        if not bsym.startswith("NCSK"):
            continue
        disp = (c.get("displayName") or bsym).replace("-USDT", "").strip().upper()
        clean = f"{disp}USDT"
        if clean not in _SYM_TO_BINGX or c.get("status") == 1:
            _SYM_TO_BINGX[clean] = bsym
            _BINGX_TO_SYM[bsym] = clean
    return _CONTRACTS_CACHE


def to_bingx_symbol(sym: str) -> str:
    _ensure_contracts()
    s = sym.strip().upper()
    if s in _SYM_TO_BINGX:
        return _SYM_TO_BINGX[s]
    if s.endswith("USDT") and not s.endswith("-USDT"):
        return s[:-4] + "-USDT"
    return s


def _raw_get(path: str, params: dict | None = None, retries: int = 5) -> Any:
    last_exc: Exception | None = None
    for i in range(retries):
        try:
            r = SESSION.get(BASE + path, params=params, timeout=25)
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(0.5 * (i + 1))
                continue
            r.raise_for_status()
            js = r.json()
            if isinstance(js, dict) and js.get("code", 0) not in (0, None):
                code = js.get("code")
                if code in (100410, 109400, 80014):
                    time.sleep(0.6 * (i + 1))
                    continue
                return js
            return js
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(0.4 * (i + 1))
    raise RuntimeError(f"BingX GET {path} failed: {last_exc}")


def get_json(path: str, params: dict | None = None, retries: int = 5) -> Any:
    params = dict(params or {})
    if path == "/fapi/v1/exchangeInfo":
        contracts = _ensure_contracts()
        symbols_out = []
        for c in contracts:
            bsym = c.get("symbol") or ""
            if not bsym.startswith("NCSK"):
                continue
            if c.get("status") != 1:
                continue
            clean = _BINGX_TO_SYM.get(bsym)
            if not clean:
                continue
            symbols_out.append({
                "symbol": clean,
                "bingxSymbol": bsym,
                "quoteAsset": "USDT",
                "status": "TRADING",
                "contractType": "TRADIFI_PERPETUAL",
                "underlyingType": "EQUITY",
                "onboardDate": int(c.get("launchTime") or 0),
            })
        return {"symbols": symbols_out}

    if path == "/fapi/v1/ticker/24hr":
        _ensure_contracts()
        js = _raw_get("/openApi/swap/v2/quote/ticker", retries=retries)
        data = js.get("data") or []
        out = []
        for t in data:
            bsym = t.get("symbol") or ""
            clean = _BINGX_TO_SYM.get(bsym)
            if not clean:
                continue
            out.append({
                "symbol": clean,
                "quoteVolume": str(t.get("quoteVolume") or "0"),
                "lastPrice": str(t.get("lastPrice") or "0"),
            })
        return out

    if path == "/fapi/v1/klines":
        sym = str(params.get("symbol") or "")
        bsym = to_bingx_symbol(sym)
        interval = str(params.get("interval") or "1h")
        want = int(params.get("limit") or 1000)
        cur_end = int(params["endTime"]) if ("endTime" in params and params["endTime"] is not None) else None
        start_ms = int(params["startTime"]) if ("startTime" in params and params["startTime"] is not None) else None

        collected: Dict[int, list] = {}
        while len(collected) < want:
            batch_lim = min(1000, want - len(collected))
            q: Dict[str, Any] = {
                "symbol": bsym,
                "interval": interval,
                "limit": max(batch_lim, 200),
            }
            if start_ms is not None:
                q["startTime"] = start_ms
            if cur_end is not None:
                q["endTime"] = cur_end
            js = _raw_get("/openApi/swap/v2/quote/klines", params=q, retries=retries)
            rows = js.get("data") or []
            if not isinstance(rows, list) or not rows:
                break
            times = []
            for r in rows:
                ts = int(r["time"])
                times.append(ts)
                collected[ts] = [
                    ts,
                    str(r["open"]),
                    str(r["high"]),
                    str(r["low"]),
                    str(r["close"]),
                    str(r["volume"]),
                ]
            min_ts = min(times)
            if len(rows) < q["limit"] or (cur_end is not None and min_ts >= cur_end):
                break
            cur_end = min_ts - 1
            time.sleep(0.04)

        sorted_ts = sorted(collected.keys())
        if len(sorted_ts) > want:
            sorted_ts = sorted_ts[-want:]
        return [collected[t] for t in sorted_ts]

    raise ValueError(f"Unsupported path in bingx_fapi shim: {path}")

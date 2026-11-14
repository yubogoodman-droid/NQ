#!/usr/bin/env python3
"""把幣安 15m 公開 K 線抓到本機快取（只讀，不下單）。

研究參數時重跑很多次，不要每次都打幣安。每檔存成一個 .npz。
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import requests

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("align_scan", REPO / "examples" / "scan_binance_15m_align.py")
scan = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scan)

TZ = scan.TZ
BAR_MS = scan.BAR_MS
CACHE = Path.home() / ".cache" / "nq-klines-15m"
PER_REQ = 1000
MAX_WEIGHT = 1500


def _get(params: dict) -> list:
    delay = 2.0
    for _ in range(10):
        try:
            r = scan.SESSION.get(scan.BASE + "/fapi/v1/klines", params=params, timeout=20)
        except requests.RequestException:
            time.sleep(delay)
            delay = min(delay * 2, 60)
            continue
        used = int(r.headers.get("X-MBX-USED-WEIGHT-1M", "0") or 0)
        if r.status_code in (418, 429):
            wait = float(r.headers.get("Retry-After", "0") or 0) or delay
            time.sleep(wait)
            delay = min(delay * 2, 120)
            continue
        r.raise_for_status()
        if used > MAX_WEIGHT:
            time.sleep(15)
        return r.json()
    raise RuntimeError(f"gave up {params.get('symbol')}")


def fetch(sym: str, start_ms: int, end_ms: int) -> dict | None:
    rows: list = []
    cur = start_ms
    while cur < end_ms:
        part = _get({"symbol": sym, "interval": "15m", "limit": PER_REQ, "startTime": cur, "endTime": end_ms - 1})
        if not part:
            break
        rows.extend(part)
        nxt = int(part[-1][0]) + BAR_MS
        if nxt <= cur:
            break
        cur = nxt
        time.sleep(0.05)
    if not rows:
        return None
    now_ms = int(time.time() * 1000)
    rows = [x for x in rows if int(x[0]) + BAR_MS <= now_ms]
    d = scan.bars_from_raw(rows)
    _, idx = np.unique(d["t"], return_index=True)
    return {k: v[np.sort(idx)] for k, v in d.items()}


def path_of(sym: str, start_ms: int, end_ms: int) -> Path:
    return CACHE / f"{start_ms}_{end_ms}" / f"{sym}.npz"


def load(sym: str, start_ms: int, end_ms: int) -> dict | None:
    p = path_of(sym, start_ms, end_ms)
    if not p.exists():
        return None
    z = np.load(p)
    if "empty" in z.files:
        return {}
    return {k: z[k] for k in ("t", "o", "h", "l", "c", "v")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-08-31")
    ap.add_argument("--end", default="2026-10-31", help="不含當天，台北時間")
    args = ap.parse_args()
    start = datetime.fromisoformat(args.start).replace(tzinfo=TZ)
    end = datetime.fromisoformat(args.end).replace(tzinfo=TZ)
    s_ms, e_ms = int(start.timestamp() * 1000), int(end.timestamp() * 1000)
    syms = [s for s, _ in scan.universe(0)]
    print(f"標的 {len(syms)}  {start} → {end}", flush=True)
    t0 = time.time()
    for i, sym in enumerate(syms, 1):
        p = path_of(sym, s_ms, e_ms)
        if p.exists():
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            d = fetch(sym, s_ms, e_ms)
        except Exception as e:
            print(f"err {sym} {e}", flush=True)
            continue
        if d is None:
            np.savez_compressed(p, empty=np.array([1]))
        else:
            np.savez_compressed(p, **d)
        if i % 25 == 0:
            print(f"{i}/{len(syms)}  {time.time()-t0:.0f}s", flush=True)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

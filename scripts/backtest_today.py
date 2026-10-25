#!/usr/bin/env python3
"""
Backtest "today" (UTC) for original shadow-neckline ± 爆量.

Binance fapi is geo-blocked (451) and Vision daily zips lag (~1–2d),
so klines are pulled from BingX USDT-M swap for the same symbols.
Universe = Binance USDT-M names that cleared 50M quote volume on the
last Vision day we have cached (fallback: BingX tickers).
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

from shadow_neckline_logic import (
    RAW,
    STRICT,
    VOLUME,
    DetectParams,
    detect_at_index,
    params_dict,
    prepare_indicators,
)

CACHE = Path("/tmp/bingx_um_klines")
VISION_CACHE = Path("/tmp/binance_um_klines")
OUT_DIR = Path("/workspace/output")
BINGX_TICKERS = "https://open-api.bingx.com/openApi/swap/v2/quote/ticker"
BINGX_KLINES = "https://open-api.bingx.com/openApi/swap/v2/quote/klines"
MIN_VOLUME_USDT = 50_000_000
HORIZONS = {"15m": 3, "30m": 6, "1h": 12, "2h": 24, "4h": 48, "8h": 96, "12h": 144}

# Binance stem -> BingX symbol overrides
ALIAS = {
    "龙虾USDT": "LONGXIA-USDT",
}


def utc_today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def to_bingx(sym: str) -> str:
    """BTCUSDT / BTC/USDT / BTC-USDT -> BTC-USDT"""
    s = sym.replace("/", "").replace("-", "").upper()
    if not s.endswith("USDT"):
        s = s + "USDT"
    if s in ALIAS:
        return ALIAS[s]
    return s[:-4] + "-USDT"


def to_display(bingx_sym: str) -> str:
    base = bingx_sym.replace("-USDT", "")
    # reverse alias for display
    if bingx_sym == "LONGXIA-USDT":
        return "龙虾/USDT"
    return f"{base}/USDT"


def binance_universe_stems() -> list[str]:
    """Prefer last Vision liquid day; else empty."""
    days = sorted(
        {p.name.split("-5m-")[-1].replace(".csv", "") for p in VISION_CACHE.glob("*-5m-*.csv")}
    )
    if not days:
        return []
    day = days[-1]
    stems = []
    for p in VISION_CACHE.glob(f"*-5m-{day}.csv"):
        stem = p.name.split("-5m-")[0]
        try:
            qv = float(pd.read_csv(p)["quote_volume"].sum())
        except Exception:
            continue
        if qv >= MIN_VOLUME_USDT:
            stems.append(stem)
    return sorted(set(stems))


def bingx_ticker_map() -> dict[str, dict]:
    r = requests.get(BINGX_TICKERS, timeout=60)
    r.raise_for_status()
    data = r.json().get("data") or []
    return {t["symbol"]: t for t in data if "symbol" in t}


def fetch_klines(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    rows: list[dict] = []
    cur = start_ms
    while cur < end_ms:
        params = {
            "symbol": symbol,
            "interval": "5m",
            "startTime": cur,
            "endTime": end_ms,
            "limit": 1000,
        }
        for attempt in range(4):
            try:
                r = requests.get(BINGX_KLINES, params=params, timeout=45)
                r.raise_for_status()
                payload = r.json()
                break
            except Exception:
                if attempt == 3:
                    return pd.DataFrame()
                time.sleep(0.5 * (attempt + 1))
        else:
            return pd.DataFrame()

        if payload.get("code") not in (0, "0", None) and payload.get("data") is None:
            return pd.DataFrame()
        data = payload.get("data") or []
        if not data:
            break
        data = sorted(data, key=lambda x: int(x["time"]))
        rows.extend(data)
        last = int(data[-1]["time"])
        nxt = last + 300_000
        if nxt <= cur:
            break
        cur = nxt
        time.sleep(0.05)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.rename(
        columns={
            "time": "timestamp",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        }
    )
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").astype("int64")
    # quote_volume approx = close * volume (BingX volume is base)
    df["quote_volume"] = df["close"] * df["volume"]
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    return df


def calc_pnl(df: pd.DataFrame, signal_ts: int) -> dict:
    idxs = df.index[df["timestamp"] == signal_ts].tolist()
    if not idxs or idxs[0] + 1 >= len(df):
        return {}
    i = idxs[0]
    entry = float(df.loc[i + 1, "open"])
    out = {"entry": entry}
    for name, n in HORIZONS.items():
        j = i + n
        out[name] = (
            None if j >= len(df) else (entry - float(df.loc[j, "close"])) / entry * 100
        )
    return out


def summarize(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n": 0, "symbols": 0, "reclaim_3_rate": None, "by_horizon": {}}
    by = {}
    for h in HORIZONS:
        col = f"pnl_{h}"
        x = df[col].dropna()
        by[h] = {
            "n": int(len(x)),
            "win_rate": round(float((x > 0).mean() * 100), 1) if len(x) else None,
            "avg": round(float(x.mean()), 3) if len(x) else None,
            "median": round(float(x.median()), 3) if len(x) else None,
        }
    return {
        "n": int(len(df)),
        "symbols": int(df["symbol"].nunique()),
        "reclaim_3_rate": round(float(df["reclaim_3"].mean() * 100), 1),
        "by_horizon": by,
        "symbols_list": sorted(df["symbol"].unique().tolist()),
    }


def run_day(day: str, params: DetectParams, frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    start_ms = int(pd.Timestamp(f"{day} 00:00", tz="UTC").timestamp() * 1000)
    # partial day: up to now
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    end_ms = min(start_ms + 86_400_000, now_ms)
    signals = []
    last_report: dict[str, datetime] = {}

    for sym, df in frames.items():
        if len(df) < 250:
            continue
        close, high, low, open_, sma200, sma14, sma25, sma99, volume, timestamp, sma200_15 = (
            prepare_indicators(df)
        )
        ts = df["timestamp"].to_numpy()

        for i in range(len(df)):
            if not (start_ms <= int(ts[i]) < end_ms):
                continue
            ok, d = detect_at_index(
                close,
                high,
                low,
                open_,
                sma200,
                sma14,
                sma25,
                sma99,
                volume,
                timestamp,
                sma200_15,
                i,
                params,
            )
            if not ok:
                continue
            tdt = datetime.fromtimestamp(int(ts[i]) / 1000, tz=timezone.utc)
            prev = last_report.get(sym)
            if prev and tdt < prev + timedelta(minutes=params.cooldown_min):
                continue

            reclaim = False
            for j in range(1, 4):
                if i + j < len(df) and high[i + j] > d["line_val"]:
                    reclaim = True
                    break

            pnl = calc_pnl(df, int(ts[i]))
            row = {
                "symbol": to_display(sym),
                "time_utc": tdt.strftime("%Y-%m-%d %H:%M"),
                **d,
                "entry": pnl.get("entry"),
                "reclaim_3": reclaim,
            }
            for h in HORIZONS:
                row[f"pnl_{h}"] = pnl.get(h)
            signals.append(row)
            last_report[sym] = tdt

    return pd.DataFrame(signals)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--day", default=utc_today(), help="UTC day YYYY-MM-DD")
    args = parser.parse_args()
    day = args.day
    hist = (datetime.fromisoformat(day).date() - timedelta(days=1)).isoformat()

    CACHE.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    start_ms = int(pd.Timestamp(f"{hist} 00:00", tz="UTC").timestamp() * 1000)
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    tickers = bingx_ticker_map()
    stems = binance_universe_stems()
    targets: list[str] = []
    if stems:
        for st in stems:
            bx = to_bingx(st)
            if bx in tickers:
                targets.append(bx)
    else:
        # fallback: bingx quoteVolume ranked (looser threshold)
        ranked = sorted(
            tickers.values(),
            key=lambda t: float(t.get("quoteVolume") or 0),
            reverse=True,
        )
        targets = [
            t["symbol"]
            for t in ranked
            if float(t.get("quoteVolume") or 0) >= 10_000_000
            and t["symbol"].endswith("-USDT")
        ][:80]

    targets = sorted(set(targets))
    print(
        f"📡 Today backtest day={day} hist={hist} symbols={len(targets)} "
        f"source=BingX (Binance fapi/Vision unavailable for {day})"
    )

    frames: dict[str, pd.DataFrame] = {}

    hist_start = int(pd.Timestamp(f"{hist} 00:00", tz="UTC").timestamp() * 1000)
    hist_end = int(pd.Timestamp(f"{day} 00:00", tz="UTC").timestamp() * 1000)
    day_end = end_ms

    def one(sym: str):
        stem = sym.replace("-", "")
        df = fetch_klines(sym, start_ms, end_ms)
        if df.empty:
            return sym, df
        hist_df = df[(df["timestamp"] >= hist_start) & (df["timestamp"] < hist_end)]
        day_df = df[(df["timestamp"] >= hist_end) & (df["timestamp"] < day_end)]
        if not hist_df.empty:
            hist_df.to_csv(CACHE / f"{stem}-5m-{hist}.csv", index=False)
        if not day_df.empty:
            day_df.to_csv(CACHE / f"{stem}-5m-{day}.csv", index=False)
        return sym, df

    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = [ex.submit(one, s) for s in targets]
        for fut in as_completed(futs):
            sym, df = fut.result()
            if df.empty:
                print(f"  skip {sym} (no klines)")
                continue
            frames[sym] = df
            print(f"  ok {sym} bars={len(df)}")

    print(f"loaded {len(frames)} / {len(targets)}")

    # write companion hist note
    meta = {
        "day": day,
        "hist": hist,
        "source": "bingx_usdm_swap",
        "note": (
            "Binance Vision daily zip not published for this day yet; "
            "fapi.binance.com returns HTTP 451 in this environment. "
            "Used BingX USDT-M swap 5m klines for overlapping Binance liquid symbols."
        ),
        "asof_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "symbols_requested": len(targets),
        "symbols_loaded": len(frames),
    }

    tiers = [
        ("raw", RAW, f"shadow_neckline_raw_{day.replace('-', '')}.csv"),
        ("volume", VOLUME, f"shadow_neckline_volume_{day.replace('-', '')}.csv"),
        ("volume2", STRICT, f"shadow_neckline_volume2_{day.replace('-', '')}.csv"),
    ]

    # Also refresh the chart-facing filenames for "today"
    chart_aliases = {
        "raw": (
            OUT_DIR / "shadow_neckline_backtest_1d.csv",
            OUT_DIR / "shadow_neckline_raw_summary.json",
        ),
        "volume": (
            OUT_DIR / "shadow_neckline_balanced_1d.csv",
            OUT_DIR / "shadow_neckline_balanced_summary.json",
        ),
        "volume2": (
            OUT_DIR / "shadow_neckline_strict_1d.csv",
            OUT_DIR / "shadow_neckline_strict_summary.json",
        ),
    }

    results = {}
    for name, params, fname in tiers:
        df = run_day(day, params, frames)
        summary = summarize(df)
        payload = {
            **meta,
            "tier": name,
            "params": params_dict(params),
            "summary": summary,
        }
        out_csv = OUT_DIR / fname
        out_json = OUT_DIR / fname.replace(".csv", "_summary.json")
        df.to_csv(out_csv, index=False)
        out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        alias_csv, alias_json = chart_aliases[name]
        df.to_csv(alias_csv, index=False)
        alias_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        if name == "volume":
            # keep volume-named outputs too
            df.to_csv(OUT_DIR / "shadow_neckline_volume_1d.csv", index=False)
            (OUT_DIR / "shadow_neckline_volume_summary.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        results[name] = summary
        print(f"\n=== {name} n={summary['n']} symbols={summary['symbols']} ===")
        if not df.empty:
            cols = [
                c
                for c in ["time_utc", "symbol", "vol_ratio", "bias", "pnl_15m", "pnl_1h"]
                if c in df.columns
            ]
            print(df[cols].sort_values("time_utc").to_string(index=False))
        print("1h:", summary["by_horizon"].get("1h"))

    (OUT_DIR / f"today_meta_{day.replace('-', '')}.json").write_text(
        json.dumps({"meta": meta, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\nDone.", json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()

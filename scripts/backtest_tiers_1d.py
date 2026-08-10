"""
One-day backtest for balanced/strict shadow-neckline tiers.
Uses shared SMA99 / SMA200 proximity rules.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from shadow_neckline_logic import (
    BALANCED,
    STRICT,
    DetectParams,
    detect_at_index,
    params_dict,
    prepare_indicators,
)

CACHE = Path("/tmp/binance_um_klines")
BASE_SIG = Path("/workspace/output/shadow_neckline_backtest_1d.csv")
DAY = "2026-08-09"
HIST = "2026-08-08"
MIN_VOLUME_USDT = 50_000_000
HORIZONS = {"15m": 3, "30m": 6, "1h": 12, "2h": 24, "4h": 48, "8h": 96, "12h": 144}
OUT = {
    "balanced": (
        Path("/workspace/output/shadow_neckline_balanced_1d.csv"),
        Path("/workspace/output/shadow_neckline_balanced_summary.json"),
    ),
    "strict": (
        Path("/workspace/output/shadow_neckline_strict_1d.csv"),
        Path("/workspace/output/shadow_neckline_strict_summary.json"),
    ),
}


def load_symbol(sym: str) -> pd.DataFrame:
    df = pd.concat(
        [
            pd.read_csv(CACHE / f"{sym}-5m-{HIST}.csv"),
            pd.read_csv(CACHE / f"{sym}-5m-{DAY}.csv"),
        ],
        ignore_index=True,
    )
    return df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def liquid_symbols() -> list[str]:
    syms = set()
    for p in CACHE.glob(f"*-5m-{DAY}.csv"):
        sym = p.name.split("-5m-")[0]
        if not (CACHE / f"{sym}-5m-{HIST}.csv").exists():
            continue
        qv = float(pd.read_csv(p)["quote_volume"].sum())
        if qv >= MIN_VOLUME_USDT:
            syms.add(sym)
    if BASE_SIG.exists():
        for s in pd.read_csv(BASE_SIG)["symbol"]:
            syms.add(s.replace("/", ""))
    return sorted(syms)


def calc_pnl(df: pd.DataFrame, signal_ts: int) -> dict:
    idxs = df.index[df["timestamp"] == signal_ts].tolist()
    if not idxs or idxs[0] + 1 >= len(df):
        return {}
    i = idxs[0]
    entry = float(df.loc[i + 1, "open"])
    out = {"entry": entry}
    for name, n in HORIZONS.items():
        j = i + n
        out[name] = None if j >= len(df) else (entry - float(df.loc[j, "close"])) / entry * 100
    return out


def run(params: DetectParams) -> pd.DataFrame:
    start_ms = int(pd.Timestamp(f"{DAY} 00:00", tz="UTC").timestamp() * 1000)
    end_ms = start_ms + 86_400_000
    signals = []
    last_report: dict[str, datetime] = {}

    for sym in liquid_symbols():
        df = load_symbol(sym)
        if len(df) < 250:
            continue
        close, high, low, open_, sma200, sma14, sma25, sma99 = prepare_indicators(df)
        ts = df["timestamp"].to_numpy()

        for i in range(len(df)):
            if not (start_ms <= ts[i] < end_ms):
                continue
            ok, d = detect_at_index(
                close, high, low, open_, sma200, sma14, sma25, sma99, i, params
            )
            if not ok:
                continue

            j = np.searchsorted(ts, ts[i] - 86_400_000, side="right") - 1
            chg24 = None
            if j >= 0 and close[j] > 0:
                chg24 = close[i] / close[j] - 1
                if chg24 > params.max_chg24:
                    continue

            tdt = datetime.fromtimestamp(ts[i] / 1000, tz=timezone.utc)
            prev = last_report.get(sym)
            if prev and tdt < prev + timedelta(minutes=params.cooldown_min):
                continue
            last_report[sym] = tdt

            pn = calc_pnl(df, int(ts[i]))
            reclaim = (
                bool(df.loc[i + 1 : i + 3, "high"].max() > d["line_val"])
                if i + 3 < len(df)
                else False
            )
            row = {
                "symbol": f"{sym[:-4]}/USDT" if sym.endswith("USDT") else sym,
                "time_utc": tdt.strftime("%Y-%m-%d %H:%M"),
                **d,
                "chg24": None if chg24 is None else round(chg24 * 100, 2),
                "entry": pn.get("entry"),
                "reclaim_3": reclaim,
            }
            for h in HORIZONS:
                row[f"pnl_{h}"] = pn.get(h)
            signals.append(row)
    return pd.DataFrame(signals)


def summarize(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n": 0}

    def stats(col):
        s = df[col].dropna()
        if s.empty:
            return None
        return {
            "n": int(len(s)),
            "win_rate": round(float((s > 0).mean() * 100), 1),
            "avg": round(float(s.mean()), 3),
            "median": round(float(s.median()), 3),
        }

    return {
        "n": int(len(df)),
        "symbols": int(df["symbol"].nunique()),
        "reclaim_3_rate": round(float(df["reclaim_3"].mean() * 100), 1),
        "by_horizon": {h: stats(f"pnl_{h}") for h in HORIZONS},
        "symbols_list": sorted(df["symbol"].unique().tolist()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["balanced", "strict", "all"], default="all")
    args = ap.parse_args()

    tiers = []
    if args.tier in ("balanced", "all"):
        tiers.append(("balanced", BALANCED))
    if args.tier in ("strict", "all"):
        tiers.append(("strict", STRICT))

    base = pd.read_csv(BASE_SIG) if BASE_SIG.exists() else None
    for name, params in tiers:
        df = run(params)
        out_csv, out_json = OUT[name]
        df.to_csv(out_csv, index=False)
        payload = {
            "day": DAY,
            "tier": name,
            "params": params_dict(params),
            "baseline_n": None if base is None else int(len(base)),
            "summary": summarize(df),
        }
        out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n=== {name} ===")
        print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
        if not df.empty and "MUBARAK" in "".join(df.symbol.astype(str)):
            print("MUBARAK:")
            print(
                df[df.symbol.str.contains("MUBARAK")][
                    ["time_utc", "dist_ma99_pct", "dist_ma200_pct", "pnl_1h"]
                ].to_string(index=False)
            )


if __name__ == "__main__":
    main()

"""
One-day backtest: original shadow-neckline vs original + 爆量.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from shadow_neckline_logic import (
    RAW,
    STRICT,
    VOLUME,
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
    "volume": (
        Path("/workspace/output/shadow_neckline_volume_1d.csv"),
        Path("/workspace/output/shadow_neckline_volume_summary.json"),
    ),
    # keep old filenames as aliases of volume (recommended)
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
        close, high, low, open_, sma200, sma14, sma25, sma99, volume, timestamp, sma200_15 = (
            prepare_indicators(df)
        )
        ts = df["timestamp"].to_numpy()

        for i in range(len(df)):
            if not (start_ms <= ts[i] < end_ms):
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

            # original script had no 24h change filter when scanning by volume list;
            # keep optional cap for compatibility
            if params.max_chg24 < 50:
                day_open = float(df.loc[df["timestamp"] >= start_ms, "open"].iloc[0])
                chg24 = (close[i] / day_open - 1.0) if day_open else 0.0
                if chg24 > params.max_chg24:
                    continue

            tdt = datetime.fromtimestamp(ts[i] / 1000, tz=timezone.utc)
            prev = last_report.get(sym)
            if prev and tdt < prev + timedelta(minutes=params.cooldown_min):
                continue

            # reclaim: any of next 3 highs back above neck
            reclaim = False
            for j in range(1, 4):
                if i + j < len(df) and high[i + j] > d["line_val"]:
                    reclaim = True
                    break

            pnl = calc_pnl(df, int(ts[i]))
            row = {
                "symbol": f"{sym[:-4]}/{sym[-4:]}" if sym.endswith("USDT") else sym,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tier",
        choices=["all", "volume", "strict", "rawcheck"],
        default="all",
    )
    args = parser.parse_args()

    tiers: list[tuple[str, DetectParams, str]] = []
    # name, params, out_key (balanced folder = 1.5× 爆量 recommended)
    if args.tier in ("all", "volume"):
        tiers.append(("volume", VOLUME, "volume"))
        tiers.append(("balanced", VOLUME, "balanced"))  # chart path alias
    if args.tier in ("all", "strict"):
        tiers.append(("strict", STRICT, "strict"))  # 2.0× 強爆量

    baseline_n = len(pd.read_csv(BASE_SIG)) if BASE_SIG.exists() else 0

    for name, params, out_key in tiers:
        df = run(params)
        summary = summarize(df)
        payload = {
            "day": DAY,
            "tier": name,
            "params": params_dict(params),
            "baseline_n": baseline_n,
            "summary": summary,
        }
        out_csv, out_json = OUT[out_key]
        df.to_csv(out_csv, index=False)
        out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"\n=== {name} (vol≥{params.min_vol_ratio}×) ===")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        cols = [
            c
            for c in ["time_utc", "symbol", "vol_ratio", "bias", "pnl_1h"]
            if c in df.columns
        ]
        if not df.empty:
            print(df[cols].sort_values("time_utc").to_string(index=False))

    if args.tier in ("all", "rawcheck"):
        raw_df = run(RAW)
        print(f"\n=== rawcheck n={len(raw_df)} baseline={baseline_n} ===")


if __name__ == "__main__":
    main()

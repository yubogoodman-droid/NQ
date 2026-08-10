"""
Strict shadow-neckline detector + one-day backtest (noise-reduced).

Filters vs original wick-break scanner:
1. CLOSE below neckline AND SMA14 (not wick-only)
2. Close break depth >= 0.8%
3. Bearish signal candle
4. Previous bar already closed below neckline (confirm)
5. Structure: 12 <= shoulder span <= 60, shoulder symmetry < 10%
6. CLOSE < SMA25 and SMA14 < SMA25 (short-term breakdown)
7. 24h change cap 60%
8. Cooldown 150 minutes
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as ta

CACHE = Path("/tmp/binance_um_klines")
BASE_SIG = Path("/workspace/output/shadow_neckline_backtest_1d.csv")
OUT_SIG = Path("/workspace/output/shadow_neckline_strict_1d.csv")
OUT_PNL = Path("/workspace/output/shadow_neckline_strict_pnl.csv")
OUT_SUMMARY = Path("/workspace/output/shadow_neckline_strict_summary.json")
DAY = "2026-08-09"
HIST = "2026-08-08"
MIN_VOLUME_USDT = 50_000_000
HORIZONS = {"15m": 3, "30m": 6, "1h": 12, "2h": 24, "4h": 48, "8h": 96, "12h": 144}


@dataclass
class StrictParams:
    min_span: int = 12
    max_span: int = 60
    shoulder_sym: float = 0.10
    min_bias: float = 0.05
    max_bias: float = 1.50
    min_close_break_pct: float = 0.008
    require_red: bool = True
    require_prev_close_below: bool = True
    require_sma25_break: bool = True
    require_sma14_lt_sma25: bool = True
    max_chg24: float = 0.60
    cooldown_min: int = 150


def detect_strict(close, high, low, open_, sma200, sma14, sma25, atr, curr_idx, p: StrictParams):
    if curr_idx + 1 < 250:
        return False, None

    window = 2
    start_i = curr_idx + 1 - 80
    end_i = curr_idx + 1 - window
    peaks = []
    for i in range(max(window, start_i), end_i):
        if i + window > curr_idx:
            break
        if close[i] == close[i - window : i + window + 1].max():
            peaks.append(i)
    if len(peaks) < 3:
        return False, None

    p1, p2, p3 = peaks[-3], peaks[-2], peaks[-1]
    h1, h2, h3 = high[p1], high[p2], high[p3]

    start_range = max(0, p2 - 48)
    if h2 < high[start_range:p2].max():
        return False, None

    s200 = sma200[p2]
    if np.isnan(s200):
        return False, None
    bias = (h2 - s200) / s200
    if bias < p.min_bias or bias > p.max_bias:
        return False, None

    span = p3 - p1
    if not (p.min_span <= span <= p.max_span):
        return False, None
    if not (h2 > h1 and h2 > h3):
        return False, None
    if abs(h1 - h3) / max(h1, h3) >= p.shoulder_sym:
        return False, None
    if curr_idx <= p3:
        return False, None

    dx = p3 - p1
    slope = (h3 - h1) / dx if dx else 0.0
    neck = h1 + slope * (curr_idx - p1)
    s14 = sma14[curr_idx]
    s25 = sma25[curr_idx]
    if np.isnan(s14) or np.isnan(s25):
        return False, None

    if not (close[curr_idx] < neck and close[curr_idx] < s14):
        return False, None
    close_break_pct = (neck - close[curr_idx]) / neck
    if close_break_pct < p.min_close_break_pct:
        return False, None
    if p.require_red and not (close[curr_idx] < open_[curr_idx]):
        return False, None
    if p.require_sma25_break and not (close[curr_idx] < s25):
        return False, None
    if p.require_sma14_lt_sma25 and not (s14 < s25):
        return False, None

    if p.require_prev_close_below:
        if curr_idx - 1 <= p3:
            return False, None
        neck_prev = h1 + slope * ((curr_idx - 1) - p1)
        if not (close[curr_idx - 1] < neck_prev):
            return False, None

    if high[curr_idx] >= h2:
        return False, None

    return True, {
        "price": float(close[curr_idx]),
        "bias": round(bias * 100, 2),
        "line_val": round(float(neck), 6),
        "sma14": round(float(s14), 6),
        "sma25": round(float(s25), 6),
        "close_break_pct": round(close_break_pct * 100, 2),
        "span": int(span),
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
        if j >= len(df):
            out[name] = None
        else:
            out[name] = (entry - float(df.loc[j, "close"])) / entry * 100
    return out


def run(params: StrictParams) -> pd.DataFrame:
    start_ms = int(pd.Timestamp(f"{DAY} 00:00", tz="UTC").timestamp() * 1000)
    end_ms = start_ms + 86_400_000
    signals = []
    last_report = {}

    for sym in liquid_symbols():
        df = load_symbol(sym)
        if len(df) < 250:
            continue
        close = df["close"].to_numpy(float)
        high = df["high"].to_numpy(float)
        low = df["low"].to_numpy(float)
        open_ = df["open"].to_numpy(float)
        sma200 = ta.sma(df["close"], 200).to_numpy(float)
        sma14 = ta.sma(df["close"], 14).to_numpy(float)
        sma25 = ta.sma(df["close"], 25).to_numpy(float)
        prev_c = np.roll(close, 1)
        prev_c[0] = close[0]
        tr = np.maximum(high - low, np.maximum(np.abs(high - prev_c), np.abs(low - prev_c)))
        atr = pd.Series(tr).rolling(14).mean().to_numpy(float)
        ts = df["timestamp"].to_numpy()

        for i in range(len(df)):
            if not (start_ms <= ts[i] < end_ms):
                continue
            ok, d = detect_strict(close, high, low, open_, sma200, sma14, sma25, atr, i, params)
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
            reclaim = bool(df.loc[i + 1 : i + 3, "high"].max() > d["line_val"]) if i + 3 < len(df) else False
            row = {
                "symbol": f"{sym[:-4]}/USDT" if sym.endswith("USDT") else sym,
                "time_utc": tdt.strftime("%Y-%m-%d %H:%M"),
                "price": d["price"],
                "bias": d["bias"],
                "line_val": d["line_val"],
                "sma14": d["sma14"],
                "sma25": d["sma25"],
                "close_break_pct": d["close_break_pct"],
                "span": d["span"],
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
    params = StrictParams()
    df = run(params)
    df.to_csv(OUT_SIG, index=False)

    pnl_rows = []
    for _, r in df.iterrows():
        for h in HORIZONS:
            v = r.get(f"pnl_{h}")
            if pd.isna(v):
                continue
            pnl_rows.append(
                {
                    "symbol": r["symbol"],
                    "time_utc": r["time_utc"],
                    "horizon": h,
                    "pnl_pct": v,
                    "entry": r.get("entry"),
                    "bias": r["bias"],
                    "close_break_pct": r["close_break_pct"],
                }
            )
    pd.DataFrame(pnl_rows).to_csv(OUT_PNL, index=False)

    base = pd.read_csv(BASE_SIG)
    base_pnl = pd.read_csv("/workspace/output/shadow_neckline_short_pnl.csv")
    b1 = base_pnl[base_pnl.horizon == "1h"]["pnl_pct"]
    summary = {
        "day": DAY,
        "params": asdict(params),
        "baseline": {
            "n": int(len(base)),
            "symbols": int(base.symbol.nunique()),
            "reclaim_3_rate": None,
            "pnl_1h_win": round(float((b1 > 0).mean() * 100), 1),
            "pnl_1h_avg": round(float(b1.mean()), 3),
            "pnl_1h_median": round(float(b1.median()), 3),
        },
        "strict": summarize(df),
    }
    OUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not df.empty:
        cols = [
            "symbol",
            "time_utc",
            "bias",
            "close_break_pct",
            "chg24",
            "pnl_15m",
            "pnl_1h",
            "pnl_4h",
            "reclaim_3",
        ]
        print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()

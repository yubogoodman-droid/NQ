#!/usr/bin/env python3
"""NQ 破底翻 MA Reclaim — 單檔（資料 + 偵測 + 回測 + HTML + Telegram）。

用法:
  python3 examples/nq_ma_reclaim.py
  python3 examples/nq_ma_reclaim.py backtest --period 8d --html report.html
  python3 examples/nq_ma_reclaim.py backtest --period 30d --pages
  python3 examples/nq_ma_reclaim.py alert
  python3 examples/nq_ma_reclaim.py alert --test
  python3 examples/nq_ma_reclaim.py alert --dry-run --once

Telegram 憑證放 tg_config.env（勿提交）:
  TELEGRAM_BOT_TOKEN=...
  TELEGRAM_CHAT_ID=...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

try:
    import requests
except ImportError:  # Telegram 才需要
    requests = None  # type: ignore

ET = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
STATE_PATH = ROOT / "tg_alert_state.json"
CONFIG_ENV = REPO_ROOT / "tg_config.env"
if not CONFIG_ENV.exists():
    CONFIG_ENV = ROOT / "tg_config.env"
PAGES_HTML = REPO_ROOT / "docs" / "nq-ma-reclaim" / "index.html"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def parse_period_days(period: str) -> Optional[int]:
    p = (period or "").strip().lower()
    if p.endswith("mo") and p[:-2].isdigit():
        return int(p[:-2]) * 30
    if p.endswith("d") and p[:-1].isdigit():
        return int(p[:-1])
    if p.endswith("w") and p[:-1].isdigit():
        return int(p[:-1]) * 7
    return None


def load_yfinance(symbol: str = "NQ=F", interval: str = "1m", period: str = "5d") -> pd.DataFrame:
    df = yf.download(symbol, interval=interval, period=period, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.title)
    return df.dropna()


def load_yahoo_intraday(
    symbol: str,
    interval: str,
    start: datetime,
    end: datetime,
    chunk_days: int = 7,
) -> pd.DataFrame:
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    chunks: List[pd.DataFrame] = []
    cur = start
    delta = timedelta(days=chunk_days)
    while cur < end:
        nxt = min(cur + delta, end)
        part = yf.download(
            symbol,
            interval=interval,
            start=cur,
            end=nxt,
            progress=False,
            auto_adjust=True,
        )
        if part is not None and len(part):
            if isinstance(part.columns, pd.MultiIndex):
                part.columns = part.columns.get_level_values(0)
            part = part.rename(columns=str.title)
            chunks.append(part)
            print(f"[data] {cur.date()} → {nxt.date()} bars={len(part)}", file=sys.stderr)
        else:
            print(f"[data] {cur.date()} → {nxt.date()} empty", file=sys.stderr)
        cur = nxt
        time.sleep(0.4)
    if not chunks:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    df = pd.concat(chunks).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df.dropna()


def load_bars(symbol: str, interval: str, period: str) -> pd.DataFrame:
    """Yahoo 1m period= 最多約 7–8 天；超過改用 7 日切片（約可回看 30 天）。"""
    days = parse_period_days(period)
    if days is not None and days > 8:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days)
        df = load_yahoo_intraday(symbol, interval, start, end, chunk_days=7)
        if not df.empty:
            return df
        print(f"[data] chunked {period} empty, fallback period download", file=sys.stderr)
    return load_yfinance(symbol, interval, period)


def to_et(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC").tz_convert(ET)
    else:
        out.index = out.index.tz_convert(ET)
    return out


def summarize_trades(trades: Sequence) -> dict:
    pnls = [float(getattr(t, "pnl_points", 0.0)) for t in trades]
    n = len(pnls)
    wins = sum(1 for p in pnls if p > 0)
    by_q: Dict[str, List[float]] = {}
    for t in trades:
        by_q.setdefault(getattr(t, "quality", "?"), []).append(float(getattr(t, "pnl_points", 0.0)))
    return {
        "count": n,
        "wins": wins,
        "win_rate": 100.0 * wins / n if n else 0.0,
        "total_points": float(sum(pnls)),
        "pnl": float(sum(pnls)),
        "n": n,
        "by_quality": {
            q: {
                "n": len(v),
                "wins": sum(1 for p in v if p > 0),
                "pnl": float(sum(v)),
            }
            for q, v in sorted(by_q.items())
        },
    }


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


@dataclass
class Signal:
    break_idx: int
    entry_idx: int
    entry_price: float
    stop_price: float
    target_price: float
    break_low: float
    two_hr_low: float
    w_left: float
    w_bottom: float
    w_type: str
    ma5: float
    ma10: float
    ma20: float
    ma30: float
    ma60: float
    ma200: float
    quality: str = "C"
    quality_score: int = 0
    m1_ma5_slope5: float = 0.0
    m1_ma60_slope5: float = 0.0
    m5_ma60_slope5: float = 0.0


@dataclass
class TradeResult:
    signal: Signal
    entry_idx: int
    exit_idx: int
    entry_price: float
    exit_price: float
    stop_price: float
    target_price: float
    pnl_points: float
    exit_reason: str
    quality: str


def sma(arr, n: int) -> np.ndarray:
    s = pd.Series(arr, dtype=float)
    return s.rolling(n, min_periods=n).mean().to_numpy(float)


def rolling_min_prev(arr, n: int) -> np.ndarray:
    s = pd.Series(arr, dtype=float)
    return s.shift(1).rolling(n, min_periods=n).min().to_numpy(float)


def _build_m5_close(df: pd.DataFrame) -> np.ndarray:
    """Map each 1m bar to the latest completed 5m close (no lookahead)."""
    close = df["Close"].astype(float)
    m5 = close.resample("5min", label="right", closed="right").last().dropna()
    out = np.full(len(df), np.nan, dtype=float)
    m5_idx = m5.index
    m5_vals = m5.to_numpy(float)
    j = 0
    for i, ts in enumerate(df.index):
        while j + 1 < len(m5_idx) and m5_idx[j + 1] <= ts:
            j += 1
        if j < len(m5_idx) and m5_idx[j] <= ts:
            out[i] = m5_vals[j]
    return out


def _build_m5_ma60_slope5(df: pd.DataFrame, slope_bars: int = 5) -> np.ndarray:
    """Map each 1m bar to latest 5m MA60 slope (no lookahead)."""
    close = df["Close"].astype(float)
    m5 = close.resample("5min", label="right", closed="right").last().dropna()
    ma60 = m5.rolling(60, min_periods=60).mean()
    slope = ma60 - ma60.shift(slope_bars)
    out = np.full(len(df), np.nan, dtype=float)
    m5_idx = slope.index
    m5_vals = slope.to_numpy(float)
    j = 0
    for i, ts in enumerate(df.index):
        while j + 1 < len(m5_idx) and m5_idx[j + 1] <= ts:
            j += 1
        if j < len(m5_idx) and m5_idx[j] <= ts:
            out[i] = m5_vals[j]
    return out


def find_w_bottom_at(low, high, break_idx, break_low):
    left = max(0, break_idx - 30)
    window = low[left : break_idx + 1]
    if len(window) == 0:
        return None
    return (
        float(high[break_idx]),
        float(break_low),
        "W头底" if break_low <= np.min(window) + 1e-9 else "none",
    )


def quality_from_slopes(m1_ma5_s5: float, m1_ma60_s5: float, m5_ma60_s5: float) -> Tuple[int, str]:
    score = 0
    if m1_ma5_s5 >= 15.0:
        score += 1
    if m1_ma60_s5 <= -8.0:
        score += 1
    if not np.isnan(m5_ma60_s5) and m5_ma60_s5 <= -8.0:
        score += 1
    if score >= 2:
        grade = "A"
    elif score == 1:
        grade = "B"
    else:
        grade = "C"
    return score, grade


def detect_signals(
    df,
    reclaim_window: int = 15,
    two_hour_bars: int = 120,
    stop_buffer: float = 15.0,
    target_r: float = 2.0,
    require_w: bool = False,
    min_break_depth: float = 10.0,
    max_entry_vol: float = 2.5,
    min_ma20_slope: float = -5.0,
    # ⑯ 貼著仍下彎/走平的 1m MA20 不進（擋 08-11 12:39）
    hug_ma20_pts: float = 16.0,
    hug_ma20_max_slope: float = 0.5,
    max_risk: float = 100.0,
    # ⑮：風險偏大時只准 QA（擋 08-05 型寬停損弱品質全損）
    max_risk_non_qa: float = 85.0,
    skip_hour_start: Optional[int] = 9,
    skip_hour_end: Optional[int] = 10,
    ma200_buffer: float = 40.0,
    ma60_buffer: float = 10.0,
    ma60_min_below: float = 6.0,
    min_ma60_slope: float = -6.0,
    ma60_slope_bars: int = 5,
    ma_lens: Tuple[int, ...] = (5, 10, 20, 30, 60, 200),
    vol_lookback: int = 20,
    ma20_slope_bars: int = 5,
    min_entry_gap: int = 15,
    pt_scale: float = 1.0,
    use_ma20_up_target: bool = True,
    ma20_up_target_r: float = 3.0,
    use_ma60_skip: bool = True,
    funnel: Optional[Dict[str, int]] = None,
) -> List[Signal]:
    s = float(pt_scale) if pt_scale and pt_scale > 0 else 1.0
    stop_buffer *= s
    min_break_depth *= s
    min_ma20_slope *= s
    hug_ma20_pts *= s
    hug_ma20_max_slope *= s
    max_risk *= s
    max_risk_non_qa *= s
    ma200_buffer *= s
    ma60_buffer *= s
    ma60_min_below *= s
    min_ma60_slope *= s

    close = df["Close"].to_numpy(float)
    low = df["Low"].to_numpy(float)
    high = df["High"].to_numpy(float)
    volume = df["Volume"].to_numpy(float)

    p5, p10, p20, p30, p60, p200 = ma_lens
    ma5 = sma(close, p5)
    ma10 = sma(close, p10)
    ma20 = sma(close, p20)
    ma30 = sma(close, p30)
    ma60 = sma(close, p60)
    ma200 = sma(close, p200)
    m5_ma60_slope5 = _build_m5_ma60_slope5(df, ma60_slope_bars)
    two_hr_low = rolling_min_prev(low, two_hour_bars)
    signals: List[Signal] = []
    last_entry = -(10**9)
    n = len(close)
    warmup = max(p200, two_hour_bars, 200)
    i = warmup
    fun = funnel if funnel is not None else {}

    def bump(key: str) -> None:
        fun[key] = fun.get(key, 0) + 1

    while i < n - 1:
        if np.isnan(two_hr_low[i]) or np.isnan(ma30[i]):
            i += 1
            continue

        support = float(two_hr_low[i])
        if low[i] >= support:
            i += 1
            continue

        bump("break")
        break_low = float(low[i])
        break_idx = i
        break_depth = support - break_low
        if break_depth < min_break_depth:
            bump("shallow")
            i += 1
            continue
        bump("deep_break")

        w_info = find_w_bottom_at(low, high, break_idx, break_low)
        if require_w and w_info is None:
            i += 1
            continue

        w_left, w_bottom, w_type = w_info if w_info else (0.0, break_low, "none")
        entered = False
        abandon_break = False

        for j in range(break_idx + 1, min(break_idx + reclaim_window + 1, n)):
            if np.isnan(ma30[j]):
                continue
            reclaimed = close[j] > ma20[j] and close[j] > ma30[j]
            bull_stack = ma5[j] > ma10[j] > ma20[j]
            if not (reclaimed and bull_stack):
                continue
            bump("reclaim_stack")
            vol_avg = np.mean(volume[max(0, j - vol_lookback) : j]) or 1.0
            if volume[j] / vol_avg > max_entry_vol:
                bump("skip_vol")
                continue
            if j >= ma20_slope_bars and (ma20[j] - ma20[j - ma20_slope_bars]) < min_ma20_slope:
                bump("skip_ma20_slope")
                continue
            # ⑯ 收盤貼著 1m MA20，且 MA20 仍下彎/走平 → 放棄這波破底
            ma20_s5 = float(ma20[j] - ma20[j - ma20_slope_bars]) if j >= ma20_slope_bars else 0.0
            if hug_ma20_pts > 0 and (close[j] - ma20[j]) < hug_ma20_pts and ma20_s5 <= hug_ma20_max_slope:
                bump("skip_hug_ma20")
                abandon_break = True
                last_entry = j
                break
            if skip_hour_start is not None and skip_hour_end is not None:
                h = df.index[j].hour
                if skip_hour_start <= h < skip_hour_end:
                    bump("skip_open_hour")
                    continue
            if j - last_entry < min_entry_gap:
                bump("skip_entry_gap")
                break
            entry = float(close[j])
            stop = break_low - stop_buffer
            risk = entry - stop
            if risk <= 0:
                bump("skip_bad_risk")
                break
            if max_risk > 0 and risk > max_risk:
                bump("skip_max_risk")
                continue
            if not np.isnan(ma200[j]):
                dist_ma200 = entry - float(ma200[j])
                if dist_ma200 <= 0 and dist_ma200 > -ma200_buffer:
                    bump("skip_ma200_hug")
                    continue
            slope5 = 0.0
            if j >= ma60_slope_bars and not np.isnan(ma60[j]) and not np.isnan(ma60[j - ma60_slope_bars]):
                slope5 = float(ma60[j]) - float(ma60[j - ma60_slope_bars])
            if use_ma60_skip and not np.isnan(ma60[j]):
                dist_ma60 = entry - float(ma60[j])
                dist_ma200 = entry - float(ma200[j]) if not np.isnan(ma200[j]) else 999.0
                p = s
                near_below = (-ma60_buffer < dist_ma60 <= -ma60_min_below) or (
                    -5.0 * p < dist_ma60 <= 0 and slope5 < -3.0 * p
                )
                skip_ma60 = False
                hard_skip_break = False
                if near_below and slope5 < -2.0 * p:
                    skip_ma60 = True
                elif 0 < dist_ma60 < 25 * p and slope5 < -0.75 * p and 0 < dist_ma200 < 25 * p:
                    skip_ma60 = True
                elif dist_ma200 < -65 * p and dist_ma60 < -39 * p and slope5 < -10.0 * p:
                    skip_ma60 = True
                elif (
                    -13.0 * p < dist_ma60 <= -10.0 * p
                    and slope5 < -7.0 * p
                    and -52.0 * p < dist_ma200 < -48.0 * p
                ):
                    skip_ma60 = True
                elif (
                    -13.0 * p < dist_ma60 <= -10.0 * p
                    and -65.0 * p < dist_ma200 < -55.0 * p
                    and slope5 < -4.5 * p
                ):
                    skip_ma60 = True
                elif (
                    dist_ma200 < -90.0 * p
                    and -32.0 * p < dist_ma60 < -28.0 * p
                    and slope5 < -6.0 * p
                    and df.index[j].hour >= 15
                ):
                    skip_ma60 = True
                elif (
                    dist_ma200 < -90.0 * p
                    and dist_ma60 < -25.0 * p
                    and slope5 < -10.0 * p
                    and df.index[j].hour < 6
                ):
                    skip_ma60 = True
                elif (
                    -90.0 * p < dist_ma200 < -65.0 * p
                    and dist_ma60 < -27.0 * p
                    and slope5 < -5.0 * p
                ):
                    skip_ma60 = True
                elif (
                    -55.0 * p < dist_ma200 < -35.0 * p
                    and -22.0 * p < dist_ma60 < -15.0 * p
                    and -4.0 * p < slope5 < -1.5 * p
                ):
                    skip_ma60 = True
                elif -50.0 * p < dist_ma200 < -35.0 * p and dist_ma60 > 0 and slope5 < -5.0 * p:
                    skip_ma60 = True
                elif not np.isnan(m5_ma60_slope5[j]) and m5_ma60_slope5[j] > 0 and slope5 > -5.0 * p:
                    skip_ma60 = True
                elif (
                    j >= ma20_slope_bars
                    and not np.isnan(ma20[j - ma20_slope_bars])
                    and float(ma20[j] - ma20[j - ma20_slope_bars]) < 0
                    and dist_ma60 < -30.0 * p
                    and slope5 < -10.0 * p
                    and (j - break_idx) >= 14
                ):
                    skip_ma60 = True
                    hard_skip_break = True
                    last_entry = j
                elif -50.0 * p < dist_ma200 < -35.0 * p and slope5 > -2.0 * p:
                    skip_ma60 = True
                elif (
                    -22.0 * p < dist_ma60 <= -14.0 * p
                    and -90.0 * p < dist_ma200 < -55.0 * p
                    and slope5 < -2.0 * p
                ):
                    skip_ma60 = True
                    hard_skip_break = True
                elif (
                    df.index[j].hour == 15
                    and -13.0 * p < dist_ma60 <= -8.0 * p
                    and dist_ma200 < -95.0 * p
                    and slope5 < -3.0 * p
                ):
                    skip_ma60 = True
                    hard_skip_break = True
                if skip_ma60:
                    bump("skip_ma60")
                    if hard_skip_break:
                        abandon_break = True
                        break
                    continue
            ma20_s5 = 0.0
            if j >= ma20_slope_bars and not np.isnan(ma20[j - ma20_slope_bars]):
                ma20_s5 = float(ma20[j] - ma20[j - ma20_slope_bars])
            use_r = ma20_up_target_r if (use_ma20_up_target and ma20_s5 > 0) else target_r
            target = entry + risk * use_r
            m1_ma5_s5 = 0.0
            if j >= ma60_slope_bars and not np.isnan(ma5[j - ma60_slope_bars]):
                m1_ma5_s5 = float(ma5[j] - ma5[j - ma60_slope_bars])
            m5_s5 = float(m5_ma60_slope5[j]) if not np.isnan(m5_ma60_slope5[j]) else float("nan")
            q_score, q_grade = quality_from_slopes(m1_ma5_s5, slope5, m5_s5)
            # ⑮ 寬停損只做 QA
            if max_risk_non_qa > 0 and risk > max_risk_non_qa and q_score < 2:
                bump("skip_wide_risk")
                continue
            bump("taken")
            signals.append(
                Signal(
                    break_idx,
                    j,
                    entry,
                    stop,
                    target,
                    break_low,
                    support,
                    w_left,
                    w_bottom,
                    w_type,
                    float(ma5[j]),
                    float(ma10[j]),
                    float(ma20[j]),
                    float(ma30[j]),
                    float(ma60[j]),
                    float(ma200[j]),
                    quality=q_grade,
                    quality_score=q_score,
                    m1_ma5_slope5=m1_ma5_s5,
                    m1_ma60_slope5=float(slope5),
                    m5_ma60_slope5=m5_s5 if not np.isnan(m5_s5) else 0.0,
                )
            )
            last_entry = j
            entered = True
            i = j + 5
            break

        if abandon_break:
            i = break_idx + reclaim_window + 1
            continue
        if not entered:
            i = break_idx + 1

    return signals


def simulate(
    df,
    signals: List[Signal],
    max_hold: int = 60,
    stop_on_m5_close: bool = True,
    be_after_r: float = 0.70,
    trail_after_r: float = 1.5,
    trail_lock_r: float = 0.5,
    preopen_flat: bool = True,
    use_ma20_time_exit: bool = True,
    hard_cap_bars: int = 390,
    ma_exit_period: int = 20,
    use_ma200_tp: bool = True,
    ma200_tp_lo: float = 40.0,
    ma200_tp_hi: float = 55.0,
    ma200_tp_d60: float = 18.0,
    use_ma200_near_tp: bool = True,
    ma200_near_over_lo: float = 0.0,
    ma200_near_over_hi: float = 15.0,
    ma200_near_pts: float = 5.0,
    use_ma60_up_stop: bool = True,
    ma60_up_near: float = 20.0,
    ma60_up_min_gap: float = 25.0,
    ma60_up_buffer: float = 5.0,
    ma60_slope_bars: int = 5,
) -> List[TradeResult]:
    close = df["Close"].to_numpy(float)
    high = df["High"].to_numpy(float)
    low = df["Low"].to_numpy(float)
    ma60 = sma(close, 60)
    ma200 = sma(close, 200)
    ma_exit = sma(close, ma_exit_period) if use_ma20_time_exit else None
    m5_close = _build_m5_close(df) if stop_on_m5_close else None
    results: List[TradeResult] = []
    grace = max(0, int(max_hold))
    hard_cap = max(grace, int(hard_cap_bars)) if use_ma20_time_exit else grace

    for sig in signals:
        entry_idx = sig.entry_idx
        entry = sig.entry_price
        stop = sig.stop_price
        target = sig.target_price
        risk = entry - stop
        if risk <= 0:
            continue
        cur_stop = stop
        mfe = 0.0
        entry_hour = df.index[entry_idx].hour
        gap200 = float(ma200[entry_idx] - entry) if not np.isnan(ma200[entry_idx]) else 0.0
        dist60e = float(entry - ma60[entry_idx]) if not np.isnan(ma60[entry_idx]) else 0.0
        use_ma200_tp_pos = (
            use_ma200_tp
            and entry < float(ma200[entry_idx])
            and ma200_tp_lo <= gap200 <= ma200_tp_hi
            and gap200 < 2.0 * risk
            and dist60e <= -ma200_tp_d60
        )
        tgt_over_200 = (target - float(ma200[entry_idx])) if not np.isnan(ma200[entry_idx]) else 999.0
        use_ma200_near_pos = (
            use_ma200_near_tp
            and not np.isnan(ma200[entry_idx])
            and entry < float(ma200[entry_idx])
            and ma200_near_over_lo <= tgt_over_200 <= ma200_near_over_hi
        )

        use_ma60_stop = False
        if (
            use_ma60_up_stop
            and entry_idx >= ma60_slope_bars
            and not np.isnan(ma60[entry_idx])
            and not np.isnan(ma60[entry_idx - ma60_slope_bars])
        ):
            slope60 = float(ma60[entry_idx] - ma60[entry_idx - ma60_slope_bars])
            gap_struct = float(ma60[entry_idx] - stop)
            if slope60 > 0 and abs(dist60e) <= ma60_up_near and gap_struct > ma60_up_min_gap:
                use_ma60_stop = True
                alt = float(ma60[entry_idx]) - ma60_up_buffer
                cur_stop = max(stop, min(entry, alt))

        limit = min(entry_idx + hard_cap, len(df) - 1)
        exit_idx = limit
        exit_price = float(close[exit_idx])
        exit_reason = "hardcap" if use_ma20_time_exit else "timeout"

        for k in range(entry_idx + 1, limit + 1):
            mfe = max(mfe, float(high[k] - entry))
            if be_after_r > 0 and mfe / risk >= be_after_r:
                cur_stop = max(cur_stop, entry)
            if trail_after_r > 0 and mfe / risk >= trail_after_r:
                cur_stop = max(cur_stop, entry + trail_lock_r * risk)

            if use_ma60_stop and not np.isnan(ma60[k]):
                alt = float(ma60[k]) - ma60_up_buffer
                cur_stop = max(cur_stop, min(entry, alt))

            et_h = df.index[k].hour
            et_m = df.index[k].minute
            if preopen_flat and entry_hour < 9 and (et_h > 9 or (et_h == 9 and et_m >= 30)):
                exit_idx, exit_price, exit_reason = k, float(close[k]), "preopen_flat"
                break

            stop_hit = (
                (m5_close[k] <= cur_stop)
                if stop_on_m5_close and not np.isnan(m5_close[k])
                else (low[k] <= cur_stop)
            )
            if stop_hit:
                reason = "ma60_stop" if use_ma60_stop and cur_stop > stop + 1e-9 else "stop"
                exit_idx, exit_price, exit_reason = k, float(cur_stop), reason
                break
            if use_ma200_tp_pos and not np.isnan(ma200[k]) and high[k] >= ma200[k]:
                exit_idx, exit_price, exit_reason = k, float(ma200[k]), "ma200_tp"
                break
            if (
                use_ma200_near_pos
                and not np.isnan(ma200[k])
                and high[k] >= float(ma200[k]) - ma200_near_pts
            ):
                exit_idx, exit_price, exit_reason = (
                    k,
                    min(float(high[k]), float(ma200[k])),
                    "ma200_near",
                )
                break
            if high[k] >= target:
                exit_idx, exit_price, exit_reason = k, float(target), "target"
                break

            held = k - entry_idx
            if use_ma20_time_exit:
                if (
                    held >= grace
                    and ma_exit is not None
                    and not np.isnan(ma_exit[k])
                    and float(close[k]) < float(ma_exit[k])
                ):
                    exit_idx, exit_price, exit_reason = k, float(close[k]), f"ma{ma_exit_period}"
                    break
            elif held >= grace:
                exit_idx, exit_price, exit_reason = k, float(close[k]), "timeout"
                break

        pnl = exit_price - entry
        results.append(
            TradeResult(
                signal=sig,
                entry_idx=entry_idx,
                exit_idx=exit_idx,
                entry_price=entry,
                exit_price=exit_price,
                stop_price=stop,
                target_price=target,
                pnl_points=float(pnl),
                exit_reason=exit_reason,
                quality=sig.quality,
            )
        )
    return results


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------


MA_COLORS = {
    5: "#ffa726",
    10: "#ffeb3b",
    20: "#66bb6a",
    30: "#26a69a",
    60: "#42a5f5",
    200: "#ab47bc",
}


def _equity_svg(pnls: List[float], width: int = 720, height: int = 180) -> str:
    if not pnls:
        return "<p class='muted'>no trades</p>"
    eq = np.cumsum(pnls)
    xs = np.linspace(0, width, len(eq) + 1)
    ys_src = np.concatenate([[0.0], eq])
    ymin, ymax = float(ys_src.min()), float(ys_src.max())
    pad = max(1.0, (ymax - ymin) * 0.12)
    ymin -= pad
    ymax += pad
    span = ymax - ymin or 1.0

    def yv(v: float) -> float:
        return height - (v - ymin) / span * height

    pts = " ".join(f"{xs[i]:.1f},{yv(ys_src[i]):.1f}" for i in range(len(ys_src)))
    zero = yv(0.0)
    color = "#16a34a" if eq[-1] >= 0 else "#dc2626"
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" style="background:#0f172a;border-radius:8px">'
        f'<line x1="0" y1="{zero:.1f}" x2="{width}" y2="{zero:.1f}" stroke="#334155" stroke-dasharray="4 4"/>'
        f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{pts}"/>'
        f"</svg>"
    )


def _trade_window(df: pd.DataFrame, trade: TradeResult) -> tuple[int, int]:
    start = max(0, trade.signal.break_idx - 25)
    end = min(len(df) - 1, trade.exit_idx + 18)
    return start, end


def draw_trade_png(df: pd.DataFrame, trade: TradeResult, path: Path, trade_no: int) -> Path:
    """Static 1m candle + MA card. MAs are computed on the full series (no window lookahead)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import Rectangle

    for fp in (
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    ):
        if Path(fp).exists():
            font_manager.fontManager.addfont(fp)
            plt.rcParams["font.sans-serif"] = [font_manager.FontProperties(fname=fp).get_name(), "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            break

    sig = trade.signal
    start, end = _trade_window(df, trade)
    window = df.iloc[start : end + 1]
    xs = range(len(window))
    o, h, l, c = window["Open"], window["High"], window["Low"], window["Close"]
    vol = window["Volume"] if "Volume" in window.columns else None
    close_full = df["Close"].astype(float)

    fig, (ax, axv) = plt.subplots(
        2,
        1,
        figsize=(10.4, 5.6),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1]},
        facecolor="#0c1210",
    )
    for a in (ax, axv):
        a.set_facecolor("#101814")
        a.tick_params(colors="#8aa193", labelsize=8)
        for sp in a.spines.values():
            sp.set_color("#2a3a33")

    colors_v = []
    for k in range(len(window)):
        up = float(c.iloc[k]) >= float(o.iloc[k])
        col = "#3dba7a" if up else "#e35d5d"
        ax.vlines(xs[k], float(l.iloc[k]), float(h.iloc[k]), color=col, lw=0.65)
        y0, y1 = min(float(o.iloc[k]), float(c.iloc[k])), max(float(o.iloc[k]), float(c.iloc[k]))
        if y1 == y0:
            y1 = y0 + max(float(h.iloc[k]) - float(l.iloc[k]), 1e-12) * 0.02
        ax.add_patch(Rectangle((xs[k] - 0.35, y0), 0.7, y1 - y0, facecolor=col, edgecolor=col, lw=0.25))
        colors_v.append("#3dba7a99" if up else "#e35d5d99")
    if vol is not None:
        axv.bar(list(xs), vol.astype(float), width=0.8, color=colors_v, linewidth=0)

    for n, col in MA_COLORS.items():
        ma = close_full.rolling(n, min_periods=n).mean().iloc[start : end + 1]
        ax.plot(list(xs), ma, color=col, lw=1.35 if n <= 20 else 1.05, label=f"MA{n}")

    ax.axhline(trade.stop_price, color="#e35d5d", ls=":", lw=1.0, alpha=0.85)
    ax.axhline(trade.target_price, color="#3dba7a", ls=":", lw=1.0, alpha=0.8)
    ax.axhline(sig.two_hr_low, color="#8aa193", ls="--", lw=0.85, alpha=0.55)

    bx, ex, xx = sig.break_idx - start, trade.entry_idx - start, trade.exit_idx - start
    if 0 <= bx < len(window):
        ax.scatter([bx], [sig.break_low], s=38, color="#f472b6", zorder=5)
        ax.annotate("破底", (bx, sig.break_low), textcoords="offset points", xytext=(0, -12),
                    ha="center", color="#f9a8d4", fontsize=8)
    if 0 <= ex < len(window):
        ax.axvline(ex, color="#3dba7a", ls="--", lw=0.9)
        ax.scatter([ex], [trade.entry_price], s=42, color="#00e676", marker="^", zorder=6)
    if 0 <= xx < len(window):
        ax.axvline(xx, color="#f0c14b", ls=":", lw=0.9)
        ax.scatter([xx], [trade.exit_price], s=40, color="#00c805" if trade.pnl_points > 0 else "#ff5252",
                   marker="x", zorder=6)

    et = df.index[trade.entry_idx]
    xt = df.index[trade.exit_idx]
    sign = "+" if trade.pnl_points >= 0 else ""
    ax.set_title(
        f"#{trade_no}  Q{trade.quality}  {et.strftime('%m-%d %H:%M')} → {xt.strftime('%H:%M')}  "
        f"{trade.exit_reason}  {sign}{trade.pnl_points:.1f}pt",
        color="#e8f0ea",
        fontsize=11,
    )
    ax.legend(loc="upper left", fontsize=7, frameon=False, labelcolor="#c8d5cc", ncol=6)
    step = max(1, len(window) // 6)
    ticks = list(range(0, len(window), step))
    axv.set_xticks(ticks)
    axv.set_xticklabels([window.index[i].strftime("%m-%d %H:%M") for i in ticks], color="#8aa193")
    fig.tight_layout(pad=0.45)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def _trade_img_name(df: pd.DataFrame, trade: TradeResult, trade_no: int, prefix: str = "t") -> str:
    et = df.index[trade.entry_idx]
    return f"{prefix}{trade_no:02d}_{et.strftime('%m%d_%H%M')}_q{trade.quality.lower()}.png"


def _render_trade_cards(
    df: pd.DataFrame,
    trades: List[TradeResult],
    html_path: Path,
    *,
    prefix: str = "t",
) -> str:
    cards: List[str] = []
    for i, t in enumerate(trades, 1):
        et = df.index[t.entry_idx]
        xt = df.index[t.exit_idx]
        cls = "pnl-win" if t.pnl_points > 0 else ("pnl-flat" if t.pnl_points == 0 else "pnl-loss")
        risk = t.entry_price - t.stop_price
        r_mult = (t.target_price - t.entry_price) / risk if risk > 0 else 0
        reason_cls = {
            "target": "tag-tp",
            "ma200_tp": "tag-tp",
            "ma200_near": "tag-tp",
            "stop": "tag-sl",
            "ma60_stop": "tag-sl",
        }.get(t.exit_reason, "tag-time")
        img_name = _trade_img_name(df, t, i, prefix=prefix)
        draw_trade_png(df, t, html_path.parent / "img" / img_name, i)
        chart = (
            f"<img src='img/{escape(img_name)}' alt='#{i} Q{escape(t.quality)}' "
            "style='width:100%;display:block;border-radius:10px'/>"
        )
        cards.append(
            "<article class='trade-card'>"
            "<header class='card-header'>"
            f"<div class='card-title'><span class='trade-no'>#{i} · Q{escape(t.quality)}</span>"
            f"<span class='trade-time'>{escape(et.strftime('%Y-%m-%d %H:%M'))} → {escape(xt.strftime('%m-%d %H:%M'))}</span></div>"
            f"<div class='card-pnl {cls}'>{t.pnl_points:+.1f} pts</div>"
            "</header>"
            "<div class='tags'>"
            f"<span class='tag {reason_cls}'>{escape(t.exit_reason)}</span>"
            f"<span class='tag tag-info'>1m</span>"
            f"<span class='tag tag-info'>Q{escape(t.quality)}</span>"
            "</div>"
            "<pre class='trade-detail'>"
            f"entry {t.entry_price:.2f}\n"
            f"stop  {t.stop_price:.2f}  (−{risk:.1f} pts)\n"
            f"target {t.target_price:.2f}  ({r_mult:.1f}R)\n"
            f"exit  {t.exit_price:.2f}  {t.exit_reason}\n"
            f"破底 {t.signal.break_low:.2f} / 2h低 {t.signal.two_hr_low:.2f}\n"
            f"MA5 {t.signal.ma5:.1f} / MA20 {t.signal.ma20:.1f} / MA60 {t.signal.ma60:.1f} / MA200 {t.signal.ma200:.1f}"
            "</pre>"
            f"<div class='mini-chart'>{chart}</div>"
            "</article>"
        )
    return "".join(cards)


def write_html_report(
    path: str | Path,
    df: pd.DataFrame,
    trades: List[TradeResult],
    symbol: str,
    period: str,
    funnel: Optional[Dict[str, int]] = None,
    extra_trades: Optional[List[TradeResult]] = None,
    extra_title: str = "",
) -> Path:
    stats = summarize_trades(trades)
    pnls = [t.pnl_points for t in trades]
    q_bits = []
    for q, info in stats.get("by_quality", {}).items():
        q_bits.append(f"Q{q} {info['n']}筆 {info['pnl']:+.1f}")
    q_line = " · ".join(q_bits) if q_bits else "無品質分組"
    out = Path(path)
    cards = _render_trade_cards(df, trades, out, prefix="t")
    extra_html = ""
    if extra_trades:
        extra_stats = summarize_trades(extra_trades)
        extra_cls = "pnl-win" if extra_stats["total_points"] >= 0 else "pnl-loss"
        extra_html = (
            f"<section class='summary'><h1>{escape(extra_title or '核心對照')}</h1>"
            f"<p class='muted'>關掉 hug MA20、MA60 特例、寬停損只做 QA。仍要破 2h 低、15 根內收復 MA20/30、5/10/20 多頭。</p>"
            f"<div class='cards'><div class='card'>筆數<b>{extra_stats['count']}</b></div>"
            f"<div class='card'>勝率<b>{extra_stats['win_rate']:.1f}%</b></div>"
            f"<div class='card'>總點數<b class='{extra_cls}'>{extra_stats['total_points']:+.1f}</b></div>"
            f"<div class='card'>勝/負<b>{extra_stats['wins']}/{extra_stats['count']-extra_stats['wins']}</b></div></div>"
            f"<div class='equity'>{_equity_svg([t.pnl_points for t in extra_trades])}</div></section>"
            + (_render_trade_cards(df, extra_trades, out, prefix="c") or "<div class='empty'>無交易</div>")
        )
    funnel_line = ""
    if funnel:
        funnel_line = (
            f"<p class='muted'>漏斗：破底 {funnel.get('break', 0)} → "
            f"深度夠 {funnel.get('deep_break', 0)} → "
            f"收復+排列 {funnel.get('reclaim_stack', 0)} → "
            f"進場 {funnel.get('taken', 0)}"
            f"（hug {funnel.get('skip_hug_ma20', 0)} · MA60 {funnel.get('skip_ma60', 0)} · "
            f"9點檔 {funnel.get('skip_open_hour', 0)} · 量能 {funnel.get('skip_vol', 0)} · "
            f"風險 {funnel.get('skip_max_risk', 0)} · 寬停損 {funnel.get('skip_wide_risk', 0)}）</p>"
        )

    start = df.index[0].strftime("%Y-%m-%d %H:%M")
    end = df.index[-1].strftime("%Y-%m-%d %H:%M")
    total_cls = "pnl-win" if stats["total_points"] >= 0 else "pnl-loss"
    html = f"""<!DOCTYPE html>
<html lang="zh-Hant"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"/>
<title>{escape(symbol)} 破底翻 MA Reclaim</title>
<style>
*{{box-sizing:border-box}}
body{{margin:0;background:#0b0e11;color:#e6edf3;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Noto Sans TC",sans-serif}}
.page{{max-width:560px;margin:0 auto;padding:14px 12px 32px}}
h1{{font-size:18px;margin:0 0 6px}}
.muted{{color:#8b949e;font-size:13px;line-height:1.5}}
.summary{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:14px 16px;margin-bottom:14px}}
.cards{{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0}}
.card{{background:#0d1117;padding:10px 12px;border-radius:10px;min-width:96px;border:1px solid #21262d}}
.card b{{display:block;font-size:20px;margin-top:4px}}
.equity{{margin:10px 0 4px}}
.trade-card{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:14px 14px 10px;margin-bottom:14px;overflow:hidden}}
.card-header{{display:flex;justify-content:space-between;gap:10px;margin-bottom:8px}}
.trade-no{{font-size:15px;font-weight:700}}
.trade-time{{font-size:12px;color:#8b949e}}
.card-pnl{{font-size:16px;font-weight:700;white-space:nowrap}}
.pnl-win{{color:#00c805}} .pnl-loss{{color:#ff5252}} .pnl-flat{{color:#8b949e}}
.tags{{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:10px}}
.tag{{font-size:11px;font-weight:600;padding:3px 8px;border-radius:999px;border:1px solid transparent}}
.tag-tp{{background:rgba(0,200,5,0.15);color:#3ddc68;border-color:rgba(0,200,5,0.35)}}
.tag-sl{{background:rgba(255,82,82,0.15);color:#ff7b72;border-color:rgba(255,82,82,0.35)}}
.tag-time{{background:rgba(255,193,7,0.12);color:#f0c14b;border-color:rgba(255,193,7,0.3)}}
.tag-info{{background:rgba(88,166,255,0.12);color:#79c0ff;border-color:rgba(88,166,255,0.28)}}
.trade-detail{{margin:0 0 10px;padding:10px 12px;background:#0d1117;border-radius:10px;border:1px solid #21262d;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;line-height:1.55;color:#c9d1d9;white-space:pre-wrap}}
.mini-chart{{margin:0 -6px -4px;border-radius:10px;overflow:hidden}}
.empty{{text-align:center;color:#8b949e;padding:40px 16px;background:#161b22;border-radius:14px;border:1px solid #30363d}}
</style></head><body>
<div class="page">
<section class="summary">
<h1>{escape(symbol)} 破底翻 MA Reclaim</h1>
<p class="muted">{escape(period)} · {escape(start)} → {escape(end)} ET · bars={len(df)}</p>
<div class="cards">
<div class="card">筆數<b>{stats['count']}</b></div>
<div class="card">勝率<b>{stats['win_rate']:.1f}%</b></div>
<div class="card">總點數<b class="{total_cls}">{stats['total_points']:+.1f}</b></div>
<div class="card">勝/負<b>{stats['wins']}/{stats['count']-stats['wins']}</b></div>
</div>
<p class="muted">{escape(q_line)}</p>
{funnel_line}
<div class="equity">{_equity_svg(pnls)}</div>
</section>
{cards or "<div class='empty'>無交易</div>"}
{extra_html}
</div>
</body></html>
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------


def load_dotenv(path: Path = CONFIG_ENV) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(name, default)
    return v if v not in (None, "") else default


def tg_send(token: str, chat_id: str, text: str, dry_run: bool = False) -> bool:
    if dry_run:
        print("[dry-run]\n" + text)
        return True
    if requests is None:
        print("pip install requests", file=sys.stderr)
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    r = requests.post(
        url,
        json={
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        },
        timeout=30,
    )
    if not r.ok:
        print(f"[tg] HTTP {r.status_code}: {r.text[:300]}", file=sys.stderr)
        return False
    data = r.json()
    if not data.get("ok"):
        print(f"[tg] API error: {data}", file=sys.stderr)
        return False
    return True


def load_state() -> Dict[str, Any]:
    if not STATE_PATH.exists():
        return {"alerted_entries": [], "alerted_exits": []}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"alerted_entries": [], "alerted_exits": []}


def save_state(state: Dict[str, Any]) -> None:
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _ts_et(ts):
    if getattr(ts, "tzinfo", None) is None:
        return ts.tz_localize("UTC").tz_convert(ET)
    return ts.tz_convert(ET)


def entry_key(df, sig: Signal) -> str:
    ts = _ts_et(df.index[sig.entry_idx])
    return f"{ts.isoformat()}|{sig.entry_price:.2f}"


def exit_key(df, tr: TradeResult) -> str:
    et = _ts_et(df.index[tr.entry_idx])
    xt = _ts_et(df.index[tr.exit_idx])
    return f"{et.isoformat()}->{xt.isoformat()}|{tr.exit_reason}|{tr.pnl_points:.2f}"


def fmt_entry(df, sig: Signal) -> str:
    ts = _ts_et(df.index[sig.entry_idx])
    br = _ts_et(df.index[sig.break_idx])
    risk = sig.entry_price - sig.stop_price
    r_mult = (sig.target_price - sig.entry_price) / risk if risk > 0 else 0
    last = float(df["Close"].iloc[-1])
    return (
        f"🟢 <b>破底翻進場</b>\n"
        f"時間: <code>{ts.strftime('%Y-%m-%d %H:%M')} ET</code>\n"
        f"品質: <b>Q{sig.quality}</b> ({sig.quality_score}/3)\n"
        f"進場: <code>{sig.entry_price:.2f}</code>\n"
        f"停損: <code>{sig.stop_price:.2f}</code> (−{risk:.1f} pts)\n"
        f"目標: <code>{sig.target_price:.2f}</code> ({r_mult:.1f}R)\n"
        f"破底: <code>{br.strftime('%H:%M')}</code> low={sig.break_low:.2f}\n"
        f"現價: <code>{last:.2f}</code>\n"
        f"#破底翻 #NQ #Q{sig.quality}"
    )


def fmt_exit(df, tr: TradeResult) -> str:
    et = _ts_et(df.index[tr.entry_idx])
    xt = _ts_et(df.index[tr.exit_idx])
    emoji = "🟢" if tr.pnl_points > 0 else ("⚪" if tr.pnl_points == 0 else "🔴")
    return (
        f"{emoji} <b>破底翻出場</b>\n"
        f"進場: <code>{et.strftime('%m-%d %H:%M')}</code> @ {tr.entry_price:.2f}\n"
        f"出場: <code>{xt.strftime('%m-%d %H:%M')}</code> @ {tr.exit_price:.2f}\n"
        f"原因: <b>{tr.exit_reason}</b>\n"
        f"盈虧: <b>{tr.pnl_points:+.1f} pts</b> · Q{tr.quality}\n"
        f"#破底翻 #出場"
    )


def scan_once(
    token: str,
    chat_id: str,
    *,
    dry_run: bool,
    alert_exits: bool,
    seed_alert: bool,
    lookback_hours: float,
    period: str = "5d",
) -> None:
    df = to_et(load_yfinance("NQ=F", "1m", period))
    sigs = detect_signals(df)
    trades = simulate(df, sigs)
    state = load_state()
    alerted_e: Set[str] = set(state.get("alerted_entries") or [])
    alerted_x: Set[str] = set(state.get("alerted_exits") or [])
    now = datetime.now(ET)
    cutoff = now.timestamp() - lookback_hours * 3600
    first_run = not STATE_PATH.exists() or (not alerted_e and not state.get("initialized"))

    new_entries = []
    for sig in sigs:
        k = entry_key(df, sig)
        ts = _ts_et(df.index[sig.entry_idx])
        if ts.timestamp() < cutoff:
            alerted_e.add(k)
            continue
        if k in alerted_e:
            continue
        new_entries.append((k, sig, ts))

    if first_run and not seed_alert:
        for k, _, _ in new_entries:
            alerted_e.add(k)
        for tr in trades:
            alerted_x.add(exit_key(df, tr))
        state["alerted_entries"] = sorted(alerted_e)[-200:]
        state["alerted_exits"] = sorted(alerted_x)[-200:]
        state["initialized"] = True
        state["last_scan"] = now.isoformat()
        save_state(state)
        print(
            f"[{now.strftime('%H:%M:%S')} ET] init: marked {len(new_entries)} recent signals, "
            f"bars={len(df)} last={df['Close'].iloc[-1]:.2f}"
        )
        return

    sent = 0
    for k, sig, ts in new_entries:
        ok = tg_send(token, chat_id, fmt_entry(df, sig), dry_run=dry_run)
        if ok:
            alerted_e.add(k)
            sent += 1
            print(f"[entry] {ts} Q{sig.quality} @ {sig.entry_price:.2f}")

    if alert_exits:
        for tr in trades:
            k = exit_key(df, tr)
            xt = _ts_et(df.index[tr.exit_idx])
            if xt.timestamp() < cutoff:
                alerted_x.add(k)
                continue
            if k in alerted_x:
                continue
            ok = tg_send(token, chat_id, fmt_exit(df, tr), dry_run=dry_run)
            if ok:
                alerted_x.add(k)
                sent += 1
                print(f"[exit] {xt} {tr.exit_reason} {tr.pnl_points:+.1f}")

    state["alerted_entries"] = sorted(alerted_e)[-200:]
    state["alerted_exits"] = sorted(alerted_x)[-200:]
    state["initialized"] = True
    state["last_scan"] = now.isoformat()
    save_state(state)
    print(
        f"[{now.strftime('%H:%M:%S')} ET] scan ok bars={len(df)} "
        f"sigs={len(sigs)} new_sent={sent} last={df['Close'].iloc[-1]:.2f}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


CORE_DETECT = dict(hug_ma20_pts=0.0, use_ma60_skip=False, max_risk_non_qa=0.0)


def detect_kwargs(args) -> dict:
    if getattr(args, "loose", False):
        return dict(CORE_DETECT)
    return {}


def cmd_backtest(args) -> int:
    df = to_et(load_bars(args.symbol, "1m", args.period))
    if df.empty:
        print("no data", file=sys.stderr)
        return 1
    funnel: Dict[str, int] = {}
    sigs = detect_signals(df, funnel=funnel, **detect_kwargs(args))
    trades = simulate(df, sigs)
    stats = summarize_trades(trades)
    print(f"{args.symbol} {args.period} bars={len(df)} {df.index[0]} -> {df.index[-1]}")
    print(f"trades={stats['count']} WR={stats['win_rate']:.1f}% pnl={stats['total_points']:+.1f}")
    if funnel:
        print(
            "funnel "
            f"break={funnel.get('break', 0)} deep={funnel.get('deep_break', 0)} "
            f"reclaim={funnel.get('reclaim_stack', 0)} taken={funnel.get('taken', 0)} "
            f"hug={funnel.get('skip_hug_ma20', 0)} ma60={funnel.get('skip_ma60', 0)} "
            f"hour={funnel.get('skip_open_hour', 0)} vol={funnel.get('skip_vol', 0)} "
            f"risk={funnel.get('skip_max_risk', 0)} wide={funnel.get('skip_wide_risk', 0)}"
        )
    for q, info in stats.get("by_quality", {}).items():
        print(f"  Q{q}: n={info['n']} wins={info['wins']} pnl={info['pnl']:+.1f}")
    for i, t in enumerate(trades, 1):
        print(
            f"[{i}] Q{t.quality} {df.index[t.entry_idx].strftime('%m-%d %H:%M')} "
            f"-> {df.index[t.exit_idx].strftime('%m-%d %H:%M')} "
            f"{t.exit_reason} {t.pnl_points:+.1f}"
        )

    extra_trades: List[TradeResult] = []
    extra_funnel: Dict[str, int] = {}
    if getattr(args, "pages", False) and not getattr(args, "loose", False):
        core_sigs = detect_signals(df, funnel=extra_funnel, **CORE_DETECT)
        extra_trades = simulate(df, core_sigs)
        extra_stats = summarize_trades(extra_trades)
        print(
            f"core  trades={extra_stats['count']} WR={extra_stats['win_rate']:.1f}% "
            f"pnl={extra_stats['total_points']:+.1f}  "
            f"(no hug / no MA60 specials / no wide-risk QA gate)"
        )
        for i, t in enumerate(extra_trades, 1):
            print(
                f"  [core {i}] Q{t.quality} {df.index[t.entry_idx].strftime('%m-%d %H:%M')} "
                f"-> {df.index[t.exit_idx].strftime('%m-%d %H:%M')} "
                f"{t.exit_reason} {t.pnl_points:+.1f}"
            )

    html_path = args.html
    if getattr(args, "pages", False):
        html_path = html_path or str(PAGES_HTML)
    if html_path:
        out = write_html_report(
            html_path,
            df,
            trades,
            args.symbol,
            args.period,
            funnel=funnel,
            extra_trades=extra_trades,
            extra_title="核心（關掉 hug / MA60 特例 / 寬停損 QA 門檻）",
        )
        print(f"html={out}")
    return 0


def cmd_alert(args) -> int:
    load_dotenv()
    token = env("TELEGRAM_BOT_TOKEN")
    chat_id = env("TELEGRAM_CHAT_ID")
    if not args.dry_run and (not token or not chat_id):
        print("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID (see tg_config.env.example)", file=sys.stderr)
        return 2

    if args.test:
        ok = tg_send(
            token or "",
            chat_id or "",
            f"✅ MA Reclaim bot test\n{datetime.now(ET).strftime('%Y-%m-%d %H:%M:%S')} ET",
            dry_run=args.dry_run,
        )
        return 0 if ok else 1

    print(
        f"MA Reclaim TG | interval={args.interval}s | exits={not args.no_exits} | "
        f"dry_run={args.dry_run} | lookback={args.lookback_hours}h"
    )
    while True:
        try:
            scan_once(
                token or "",
                chat_id or "",
                dry_run=args.dry_run,
                alert_exits=not args.no_exits,
                seed_alert=args.seed_alert,
                lookback_hours=args.lookback_hours,
                period=args.period,
            )
        except Exception as e:
            print(f"[error] {e}", file=sys.stderr)
            traceback.print_exc()
        if args.once:
            break
        time.sleep(max(15, args.interval))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NQ 破底翻 MA Reclaim（單檔）")
    sub = p.add_subparsers(dest="cmd")

    b = sub.add_parser("backtest", help="Yahoo 1m 回測")
    b.add_argument("--symbol", default="NQ=F")
    b.add_argument("--period", default="8d")
    b.add_argument("--html", default="")
    b.add_argument("--pages", action="store_true", help="寫到 docs/nq-ma-reclaim/index.html")
    b.add_argument("--loose", action="store_true", help="關掉 hug / MA60 特例 / 寬停損 QA，只留核心破底翻")
    b.set_defaults(func=cmd_backtest)

    a = sub.add_parser("alert", help="Telegram 輪詢")
    a.add_argument("--interval", type=int, default=None)
    a.add_argument("--once", action="store_true")
    a.add_argument("--dry-run", action="store_true")
    a.add_argument("--test", action="store_true")
    a.add_argument("--no-exits", action="store_true")
    a.add_argument("--seed-alert", action="store_true")
    a.add_argument("--lookback-hours", type=float, default=None)
    a.add_argument("--period", default="5d")
    a.set_defaults(func=cmd_alert)

    # 無子命令時當 backtest
    p.add_argument("--symbol", default="NQ=F")
    p.add_argument("--period", default="8d")
    p.add_argument("--html", default="")
    p.add_argument("--pages", action="store_true", help="寫到 docs/nq-ma-reclaim/index.html")
    p.add_argument("--loose", action="store_true", help="關掉 hug / MA60 特例 / 寬停損 QA，只留核心破底翻")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "alert":
        if args.interval is None:
            args.interval = int(env("POLL_SECONDS", "60") or 60)
        if args.lookback_hours is None:
            args.lookback_hours = float(env("LOOKBACK_HOURS", "36") or 36)
        return cmd_alert(args)
    if args.cmd is None:
        args.cmd = "backtest"
    return cmd_backtest(args)


if __name__ == "__main__":
    raise SystemExit(main())

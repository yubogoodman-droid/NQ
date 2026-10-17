#!/usr/bin/env python3
"""NQ 一分 K 高檔 M 頭：收盤跌破 MA60 做空。

用法:
  python3 examples/nq_m_head.py
  python3 examples/nq_m_head.py --period 30d --pages
  python3 examples/nq_m_head.py --period 8d --html output/nq_m_head.html
"""

from __future__ import annotations

import argparse
import base64
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from html import escape
from pathlib import Path
from typing import Dict, List, Optional, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nq.patterns import MHeadPattern, detect_m_heads

ET = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
PAGES_HTML = REPO_ROOT / "docs" / "nq-m-head" / "index.html"
POINT_VALUE = 20.0
TICK = 0.25

MA_PERIODS = (5, 10, 20, 30, 60, 120, 200)
MA_COLORS = {
    5: "#ffa726",
    10: "#ffeb3b",
    20: "#66bb6a",
    30: "#26a69a",
    60: "#42a5f5",
    120: "#26c6da",
    200: "#ab47bc",
}

# 鎖利下一根才生效，避免同一根長影線誤砍 2R。
# 1m：1.6R 鎖 1.2R（今日瀑布還在走，不要 1.0 就砍）
# 5m：多一檔 1.0R 鎖 0.7R（#2 那種砸完回補）
TRAIL_ARM_R = 1.6
TRAIL_LOCK_R = 1.2
TRAIL_STEPS_1M = ((1.6, 1.2),)
# 5m 停損較寬；第一檔 0.75R 鎖 0.5R（#8 差 ~2 點沒到舊 0.8R）
TRAIL_STEPS_5M = ((0.75, 0.5), (1.2, 0.9), (1.6, 1.2))

# 1m / 5m / 1h 同一套邏輯，K 數換成大約相同的鐘面時間
TF_PRESETS = {
    "1m": {
        "swing_lookback": 7,
        "min_bars_between_highs": 20,
        "max_bars_between_highs": 75,
        "high_level_lookback": 120,
        "entry_window": 35,
        "max_bars_hold": 120,
        "min_ribbon_spread": 28.0,
        "trail_steps": TRAIL_STEPS_1M,
    },
    "5m": {
        "swing_lookback": 3,  # 15 分鐘確認
        "min_bars_between_highs": 4,  # 20 分鐘，對得上今日 11:01–11:26 那組
        "max_bars_between_highs": 48,  # 4 小時
        "high_level_lookback": 24,  # 2 小時
        "entry_window": 16,  # 80 分鐘
        "max_bars_hold": 48,  # 4 小時
        "min_ribbon_spread": 28.0,
        "trail_steps": TRAIL_STEPS_5M,
        "stop_buffer": 36.0,  # 避開 #6 那種頭頂 +8 被軋空掃掉
        "skip_slow_sandwich": True,  # 收盤夾在 MA120/MA200 中間不空（#4 假跌破）
        "max_above_ma200": 150.0,  # #5：MA200 還在下面太遠，且 1h 已破
        "untested_htf_gap": 200.0,  # #8：破 MA200 但 1h 還沒測到、MA60 仍往上
        "ma60_slope_bars": 6,  # 5m 30 分鐘看 MA60 有沒有轉
    },
    "1h": {
        "swing_lookback": 2,  # 2 小時確認
        "min_bars_between_highs": 2,
        "max_bars_between_highs": 16,  # 16 小時
        "high_level_lookback": 8,  # 8 小時
        "entry_window": 4,
        "max_bars_hold": 16,
        "min_ribbon_spread": 28.0,
        "trail_steps": ((0.8, 0.5), (1.2, 0.9), (1.6, 1.2)),
        "stop_buffer": 50.0,
        "skip_slow_sandwich": True,
    },
}


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


def _normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    return df[keep].dropna()


def load_yfinance(symbol: str = "NQ=F", interval: str = "1m", period: str = "5d") -> pd.DataFrame:
    df = yf.download(symbol, interval=interval, period=period, progress=False, auto_adjust=True)
    return _normalize_ohlc(df)


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
            chunks.append(_normalize_ohlc(part))
            print(f"[data] {cur.date()} → {nxt.date()} bars={len(chunks[-1])}", file=sys.stderr)
        else:
            print(f"[data] {cur.date()} → {nxt.date()} empty", file=sys.stderr)
        cur = nxt
        time.sleep(0.4)
    if not chunks:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.concat(chunks).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df.dropna()


def load_bars(symbol: str, interval: str, period: str) -> pd.DataFrame:
    """Yahoo 1m 超過 8 天改 7 日切片，最多約 30 天；5m 可回看約 60 天。"""
    days = parse_period_days(period)
    if days is not None and days > 8:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days)
        if interval == "1m":
            # Yahoo 1m 實際只能回看約 30 天，起點再早會整段空掉
            min_start = end - timedelta(days=29, hours=18)
            if start < min_start:
                start = min_start
        elif interval == "5m":
            # Yahoo 5m 剛好 60 天會把第一週整段空掉
            min_start = end - timedelta(days=59, hours=20)
            if start < min_start:
                start = min_start
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


def round_tick(price: float) -> float:
    return round(price / TICK) * TICK


def sma(arr, n: int) -> np.ndarray:
    s = pd.Series(arr, dtype=float)
    return s.rolling(n, min_periods=n).mean().to_numpy(float)


def ribbon_spread(*values: float) -> float:
    """MA5/10/20/30/60 帶寬：max − min。任一缺值回傳 nan。"""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.any(np.isnan(arr)):
        return float("nan")
    return float(arr.max() - arr.min())


def ribbon_tangled(*values: float, min_spread: float) -> bool:
    """均線糾結：帶寬太窄，或算不出帶寬。"""
    spread = ribbon_spread(*values)
    return bool(np.isnan(spread) or spread < min_spread)


def slow_ma_sandwich(close: float, ma120: float, ma200: float) -> bool:
    """收盤夾在 MA120 與 MA200 之間：慢均還在托，MA60 常是假跌破。"""
    if np.isnan(close) or np.isnan(ma120) or np.isnan(ma200):
        return False
    lo, hi = (ma120, ma200) if ma120 <= ma200 else (ma200, ma120)
    return bool(lo < close < hi)


def far_above_ma200(
    close: float,
    ma200: float,
    max_gap: float,
    htf_ma: float = float("nan"),
    min_break_pts: float = 8.0,
) -> bool:
    """MA200 還在下面太遠，且 1h 已經跌破：假跌破（#5）。1h 沒破（#6）放過。"""
    if max_gap <= 0 or np.isnan(close) or np.isnan(ma200) or np.isnan(htf_ma):
        return False
    if close < ma200 + max_gap:
        return False
    return bool(close <= htf_ma - min_break_pts)


def untested_htf_support(
    close: float,
    ma60: float,
    ma60_prev: float,
    ma200: float,
    htf_ma: float,
    min_gap: float,
) -> bool:
    """已破 MA200，但 1h MA60 還在下面很遠，且 5m MA60 仍往上：砸到慢均、大週期還沒測到。"""
    if min_gap <= 0:
        return False
    if any(np.isnan(v) for v in (close, ma60, ma60_prev, ma200, htf_ma)):
        return False
    if close >= ma200:
        return False
    if close < htf_ma + min_gap:
        return False
    if ma60 <= ma60_prev:
        return False
    return True


def overlay_htf_ma60(df: pd.DataFrame, df_htf: pd.DataFrame, *, col: str) -> pd.DataFrame:
    """把已收盤的較大週期 MA60 對齊到小週期（不偷看當根未收的 HTF）。"""
    out = df.copy()
    out[col] = np.nan
    if df_htf is None or df_htf.empty:
        return out
    ma = df_htf["close"].astype(float).rolling(60, min_periods=60).mean().shift(1)
    out[col] = ma.reindex(out.index, method="ffill")
    return out


def overlay_m5_ma60(df_1m: pd.DataFrame, df_5m: pd.DataFrame) -> pd.DataFrame:
    """把已收盤的 5m MA60 對齊到 1m（不偷看當根未收的 5 分 K）。"""
    return overlay_htf_ma60(df_1m, df_5m, col="ma60_5m")


def htf_snapshot(df: pd.DataFrame, bar_idx: int, col: str, label: str) -> str:
    if col not in df.columns:
        return ""
    ma = df[col].iloc[bar_idx]
    if pd.isna(ma):
        return f"{label} n/a"
    close = float(df["close"].iloc[bar_idx])
    diff = close - float(ma)
    side = "低於" if diff < 0 else "高於"
    return f"{label} {float(ma):.2f}  收盤{side} {diff:+.1f}"


def m5_snapshot(df: pd.DataFrame, bar_idx: int) -> str:
    return htf_snapshot(df, bar_idx, "ma60_5m", "5m MA60")


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


@dataclass
class Signal:
    timestamp: pd.Timestamp
    entry: float
    stop_loss: float
    target: float
    pattern: MHeadPattern
    bar_idx: int
    ma60: float
    ma20: float
    ma5: float
    timeframe: str = "1m"
    ribbon_spread: float = 0.0

    @property
    def risk(self) -> float:
        return self.stop_loss - self.entry

    @property
    def reward(self) -> float:
        return self.entry - self.target


@dataclass
class TradeResult:
    signal: Signal
    exit_price: float
    exit_time: pd.Timestamp
    exit_idx: int
    exit_reason: str
    pnl_points: float
    pnl_dollars: float


def _is_high_level(
    df: pd.DataFrame,
    pattern: MHeadPattern,
    ma60: np.ndarray,
    *,
    lookback: int = 120,
    near_pct: float = 0.002,
) -> bool:
    """高峰需接近近 lookback 根高點；M 頭坐在 MA60 上方（頸線當下仍高於 MA60）。"""
    h2 = pattern.second_high_idx
    neck_i = pattern.neckline_idx
    if h2 >= len(ma60) or neck_i >= len(ma60):
        return False
    if np.isnan(ma60[pattern.first_high_idx]) or np.isnan(ma60[h2]) or np.isnan(ma60[neck_i]):
        return False
    if pattern.first_high < float(ma60[pattern.first_high_idx]):
        return False
    if pattern.second_high < float(ma60[h2]):
        return False
    if float(df["close"].iloc[h2]) < float(ma60[h2]):
        return False
    # 用頸線那根的 MA60，避免盤整時均線追上頸線而被誤殺
    if pattern.neckline < float(ma60[neck_i]):
        return False
    start = max(0, h2 - lookback)
    window_high = float(df["high"].iloc[start : h2 + 1].max())
    if window_high <= 0:
        return False
    return pattern.peak >= window_high * (1.0 - near_pct)


def generate_signals(
    df: pd.DataFrame,
    *,
    swing_lookback: int = 7,
    high_tolerance_pct: float = 0.0012,
    min_bars_between_highs: int = 20,
    max_bars_between_highs: int = 75,
    min_depth_pct: float = 0.0015,
    high_level_lookback: int = 120,
    high_level_pct: float = 0.0015,
    entry_window: int = 35,
    stop_buffer: float = 8.0,
    min_risk: float = 50.0,
    max_risk: float = 220.0,
    min_h2_extension: float = 30.0,
    min_break_pts: float = 8.0,
    session_start: Optional[int] = None,
    session_end: Optional[int] = None,
    target_r: float = 2.0,
    use_measured_target: bool = False,
    timeframe: str = "1m",
    min_ribbon_spread: float = 28.0,
    skip_slow_sandwich: bool = False,
    max_above_ma200: float = 0.0,
    untested_htf_gap: float = 0.0,
    ma60_slope_bars: int = 6,
    funnel: Optional[Dict[str, int]] = None,
) -> List[Signal]:
    """高檔 M 頭確認後，收盤跌破 MA60 做空；均線還糾結就先等打開。"""
    fun = funnel if funnel is not None else {}

    def bump(key: str) -> None:
        fun[key] = fun.get(key, 0) + 1

    patterns = detect_m_heads(
        df,
        swing_lookback=swing_lookback,
        high_tolerance_pct=high_tolerance_pct,
        min_bars_between_highs=min_bars_between_highs,
        max_bars_between_highs=max_bars_between_highs,
        min_depth_pct=min_depth_pct,
    )
    fun["m_heads"] = len(patterns)

    close = df["close"].to_numpy(float)
    high = df["high"].to_numpy(float)
    ma5 = sma(close, 5)
    ma10 = sma(close, 10)
    ma20 = sma(close, 20)
    ma30 = sma(close, 30)
    ma60 = sma(close, 60)
    ma120 = sma(close, 120)
    ma200 = sma(close, 200)
    htf_col = df["ma60_1h"].to_numpy(float) if "ma60_1h" in df.columns else None
    n = len(df)
    signals: List[Signal] = []
    used_entry: set[int] = set()

    for p in patterns:
        confirm = p.second_high_idx + swing_lookback
        if confirm >= n:
            bump("skip_unconfirmed")
            continue
        if not _is_high_level(
            df, p, ma60, lookback=high_level_lookback, near_pct=high_level_pct
        ):
            bump("skip_not_high")
            continue
        h2_ma = float(ma60[p.second_high_idx])
        if float(df["close"].iloc[p.second_high_idx]) - h2_ma < min_h2_extension:
            bump("skip_thin_ext")
            continue
        bump("high_level")

        end = min(n - 1, confirm + entry_window)
        invalidated = False
        entry_idx: int | None = None
        broke_structure = False
        saw_sandwich = False
        saw_far_ma200 = False
        saw_htf_gap = False
        for k in range(confirm, end + 1):
            if high[k] > p.peak:
                invalidated = True
                break
            if np.isnan(ma60[k]) or np.isnan(ma5[k]) or np.isnan(ma10[k]) or np.isnan(ma20[k]) or np.isnan(ma30[k]):
                continue
            if skip_slow_sandwich and (np.isnan(ma120[k]) or np.isnan(ma200[k])):
                continue
            # 真跌破：收盤同時低於 MA60（不是 1 點吻線）與 M 頸線。
            # 排除「價格橫盤、MA60 往上追上」——那種收盤仍在頸線之上。
            under_ma = close[k] <= ma60[k] - min_break_pts
            under_neck = close[k] < p.neckline
            if not (under_ma and under_neck):
                continue
            broke_structure = True
            # 11:00 那種 MA5/10/20/30/60 黏成一團先不進，等到帶寬打開。
            if ribbon_tangled(
                float(ma5[k]),
                float(ma10[k]),
                float(ma20[k]),
                float(ma30[k]),
                float(ma60[k]),
                min_spread=min_ribbon_spread,
            ):
                continue
            # 夾在 MA120/MA200 中間先不進；一旦進過夾心，必須收盤跌出慢均下方才算真破。
            if skip_slow_sandwich:
                lo_slow = min(float(ma120[k]), float(ma200[k]))
                if slow_ma_sandwich(float(close[k]), float(ma120[k]), float(ma200[k])):
                    saw_sandwich = True
                    continue
                if saw_sandwich and close[k] > lo_slow - min_break_pts:
                    continue
            htv = float(htf_col[k]) if htf_col is not None else float("nan")
            if far_above_ma200(float(close[k]), float(ma200[k]), max_above_ma200, htv, min_break_pts):
                saw_far_ma200 = True
                break
            if untested_htf_gap > 0 and htf_col is not None and k >= ma60_slope_bars:
                if untested_htf_support(
                    float(close[k]),
                    float(ma60[k]),
                    float(ma60[k - ma60_slope_bars]),
                    float(ma200[k]),
                    float(htf_col[k]),
                    untested_htf_gap,
                ):
                    saw_htf_gap = True
                    break
            entry_idx = k
            break
        if invalidated:
            bump("skip_invalidated")
            continue
        if entry_idx is None:
            if saw_htf_gap:
                bump("skip_htf_gap")
            elif saw_far_ma200:
                bump("skip_far_ma200")
            elif saw_sandwich:
                bump("skip_sandwich")
            elif broke_structure:
                bump("skip_tangled")
            else:
                bump("skip_no_ma60")
            continue
        if session_start is not None and session_end is not None:
            ts = df.index[entry_idx]
            if getattr(ts, "tzinfo", None):
                ts = ts.tz_convert(ET)
            minutes = int(ts.hour) * 60 + int(ts.minute)
            if not (session_start <= minutes < session_end):
                bump("skip_session")
                continue
        if entry_idx in used_entry:
            bump("skip_dup_entry")
            continue

        entry = round_tick(float(close[entry_idx]))
        stop = round_tick(p.peak + stop_buffer)
        risk = stop - entry
        if risk < min_risk:
            bump("skip_tiny_risk")
            continue
        if risk > max_risk:
            bump("skip_wide_risk")
            continue

        measured = round_tick(p.target)
        r_target = round_tick(entry - risk * target_r)
        if use_measured_target:
            target = min(measured, r_target)
        else:
            target = r_target
        if target >= entry:
            bump("skip_bad_target")
            continue

        bump("taken")
        used_entry.add(entry_idx)
        signals.append(
            Signal(
                timestamp=df.index[entry_idx],
                entry=entry,
                stop_loss=stop,
                target=target,
                pattern=p,
                bar_idx=entry_idx,
                ma60=float(ma60[entry_idx]),
                ma20=float(ma20[entry_idx]) if not np.isnan(ma20[entry_idx]) else 0.0,
                ma5=float(ma5[entry_idx]) if not np.isnan(ma5[entry_idx]) else 0.0,
                timeframe=timeframe,
                ribbon_spread=ribbon_spread(
                    float(ma5[entry_idx]),
                    float(ma10[entry_idx]),
                    float(ma20[entry_idx]),
                    float(ma30[entry_idx]),
                    float(ma60[entry_idx]),
                ),
            )
        )

    return sorted(signals, key=lambda s: s.bar_idx)


def apply_preset(timeframe: str) -> dict:
    if timeframe not in TF_PRESETS:
        raise ValueError(f"unknown timeframe {timeframe}")
    return dict(TF_PRESETS[timeframe])


def run_tf_backtest(
    df: pd.DataFrame,
    timeframe: str,
    extra: Optional[dict] = None,
) -> tuple[list[Signal], list[TradeResult], dict]:
    preset = apply_preset(timeframe)
    hold = preset.pop("max_bars_hold")
    trail_steps = preset.pop("trail_steps", TRAIL_STEPS_1M)
    funnel: Dict[str, int] = {}
    sigs = generate_signals(df, funnel=funnel, timeframe=timeframe, **preset, **(extra or {}))
    trades = run_backtest(df, sigs, max_bars_hold=hold, trail_steps=trail_steps)
    return sigs, trades, funnel


def _trail_lock_price(
    entry: float,
    risk: float,
    mfe: float,
    steps: Sequence[tuple[float, float]],
) -> float | None:
    """依最大浮盈 R 取最緊的鎖利價（做空：價越低鎖越多）。"""
    if risk <= 0 or not steps:
        return None
    r_seen = mfe / risk
    lock_r = None
    for arm_r, keep_r in steps:
        if r_seen >= arm_r:
            lock_r = keep_r if lock_r is None else max(lock_r, keep_r)
    if lock_r is None:
        return None
    return round_tick(entry - lock_r * risk)


def run_backtest(
    df: pd.DataFrame,
    signals: Sequence[Signal] | None = None,
    *,
    max_bars_hold: int = 120,
    trail_steps: Sequence[tuple[float, float]] | None = None,
    trail_arm_r: float = TRAIL_ARM_R,
    trail_lock_r: float = TRAIL_LOCK_R,
) -> List[TradeResult]:
    """做空：結構停損 → 2R → 鎖利（下一根生效）→ 逾時。不重疊。"""
    if signals is None:
        signals = generate_signals(df)
    steps = list(trail_steps) if trail_steps is not None else [(trail_arm_r, trail_lock_r)]
    results: List[TradeResult] = []
    position_open_until = -1

    for sig in signals:
        entry_idx = sig.bar_idx
        if entry_idx <= position_open_until:
            continue
        end_idx = min(entry_idx + max_bars_hold, len(df) - 1)
        exit_price = float(df["close"].iloc[end_idx])
        exit_time = df.index[end_idx]
        exit_reason = "time_stop"
        exit_idx = end_idx
        orig_stop = sig.stop_loss
        trail_stop = orig_stop
        pending_lock: float | None = None
        mfe = 0.0
        risk = sig.risk

        for i in range(entry_idx + 1, end_idx + 1):
            lo = float(df["low"].iloc[i])
            hi = float(df["high"].iloc[i])
            if pending_lock is not None:
                trail_stop = min(trail_stop, pending_lock)
                pending_lock = None
            if hi >= orig_stop:
                exit_price = orig_stop
                exit_time = df.index[i]
                exit_reason = "stop_loss"
                exit_idx = i
                break
            if lo <= sig.target:
                exit_price = sig.target
                exit_time = df.index[i]
                exit_reason = "take_profit"
                exit_idx = i
                break
            if trail_stop < orig_stop - 1e-9 and hi >= trail_stop:
                exit_price = trail_stop
                exit_time = df.index[i]
                exit_reason = "trail_stop"
                exit_idx = i
                break
            mfe = max(mfe, sig.entry - lo)
            locked = _trail_lock_price(sig.entry, risk, mfe, steps)
            if locked is not None:
                pending_lock = locked if pending_lock is None else min(pending_lock, locked)

        position_open_until = exit_idx
        pnl_points = sig.entry - exit_price
        results.append(
            TradeResult(
                signal=sig,
                exit_price=exit_price,
                exit_time=exit_time,
                exit_idx=exit_idx,
                exit_reason=exit_reason,
                pnl_points=pnl_points,
                pnl_dollars=pnl_points * POINT_VALUE,
            )
        )
    return results


def summarize(results: Sequence[TradeResult]) -> dict:
    if not results:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "total_pnl_points": 0.0,
            "total_pnl_dollars": 0.0,
            "avg_pnl_points": 0.0,
        }
    wins = sum(1 for r in results if r.pnl_points > 0)
    total = sum(r.pnl_points for r in results)
    dollars = sum(r.pnl_dollars for r in results)
    return {
        "trades": len(results),
        "wins": wins,
        "losses": len(results) - wins,
        "win_rate": wins / len(results),
        "total_pnl_points": total,
        "total_pnl_dollars": dollars,
        "avg_pnl_points": total / len(results),
    }


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------


def _fmt_time(ts: pd.Timestamp) -> str:
    t = ts.tz_convert(ET) if getattr(ts, "tzinfo", None) else ts
    return t.strftime("%m-%d %H:%M")


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


def _chart_window(df: pd.DataFrame, trade: TradeResult) -> tuple[int, int]:
    p = trade.signal.pattern
    start = max(0, p.first_high_idx - 16)
    end = min(
        len(df) - 1,
        max(trade.exit_idx + 10, trade.signal.bar_idx + 18, p.second_high_idx + 16),
    )
    return start, end


def _trade_img_name(trade: TradeResult, trade_no: int, prefix: str = "m") -> str:
    ts = trade.signal.timestamp
    if getattr(ts, "tzinfo", None):
        ts = ts.tz_convert(ET)
    return f"{prefix}{trade_no:02d}_{ts.strftime('%m%d_%H%M')}.png"


def draw_trade_png(df: pd.DataFrame, trade: TradeResult, path: Path, trade_no: int) -> Path:
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
            plt.rcParams["font.sans-serif"] = [
                font_manager.FontProperties(fname=fp).get_name(),
                "DejaVu Sans",
            ]
            plt.rcParams["axes.unicode_minus"] = False
            break

    sig = trade.signal
    p = sig.pattern
    start, end = _chart_window(df, trade)
    window = df.iloc[start : end + 1]
    xs = range(len(window))
    o, h, l, c = window["open"], window["high"], window["low"], window["close"]
    vol = window["volume"] if "volume" in window.columns else None
    close_full = df["close"].astype(float)

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

    for period in MA_PERIODS:
        ma = close_full.rolling(period, min_periods=period).mean().iloc[start : end + 1]
        if ma.notna().sum() == 0:
            continue
        lw = 1.85 if period == 60 else (1.35 if period <= 20 else 1.05)
        ax.plot(list(xs), ma, color=MA_COLORS[period], lw=lw, label=f"MA{period}")
    if "ma60_5m" in df.columns:
        m5 = df["ma60_5m"].iloc[start : end + 1]
        if m5.notna().sum():
            ax.plot(list(xs), m5, color="#f48fb1", lw=1.6, ls="--", label="5mMA60")
    if "ma60_1h" in df.columns:
        h1 = df["ma60_1h"].iloc[start : end + 1]
        if h1.notna().sum():
            ax.plot(list(xs), h1, color="#ce93d8", lw=1.6, ls="--", label="1hMA60")

    neck_start = p.first_high_idx - start
    neck_end = sig.bar_idx - start
    ax.hlines(p.neckline, max(0, neck_start), min(len(window) - 1, neck_end),
              colors="#ffa726", linestyles="--", lw=1.0, alpha=0.9)
    ax.axhline(sig.stop_loss, color="#e35d5d", ls=":", lw=1.0, alpha=0.85)
    ax.axhline(sig.target, color="#3dba7a", ls=":", lw=1.0, alpha=0.8)
    ax.axhline(sig.ma60, color="#42a5f5", ls=":", lw=0.8, alpha=0.45)

    h1_rel = p.first_high_idx - start
    h2_rel = p.second_high_idx - start
    if 0 <= h1_rel < len(window):
        ax.scatter([h1_rel], [p.first_high], s=42, color="#42a5f5", zorder=5)
        ax.annotate("H1", (h1_rel, p.first_high), textcoords="offset points", xytext=(0, 10),
                    ha="center", color="#79c0ff", fontsize=8)
    if 0 <= h2_rel < len(window):
        ax.scatter([h2_rel], [p.second_high], s=42, color="#ec407a", zorder=5)
        ax.annotate("H2", (h2_rel, p.second_high), textcoords="offset points", xytext=(0, 10),
                    ha="center", color="#f9a8d4", fontsize=8)

    entry_rel = sig.bar_idx - start
    exit_rel = trade.exit_idx - start
    if 0 <= entry_rel < len(window):
        ax.axvline(entry_rel, color="#e35d5d", ls="--", lw=0.9)
        ax.scatter([entry_rel], [sig.entry], s=48, color="#ff5252", marker="v", zorder=6)
    if 0 <= exit_rel < len(window):
        ax.axvline(exit_rel, color="#f0c14b", ls=":", lw=0.9)
        ax.scatter(
            [exit_rel],
            [trade.exit_price],
            s=44,
            color="#00c805" if trade.pnl_points > 0 else "#ff5252",
            marker="x",
            zorder=6,
        )

    y_min = float(window["low"].min())
    y_max = float(window["high"].max())
    pad = max((y_max - y_min) * 0.06, 2.0)
    ax.set_ylim(y_min - pad, y_max + pad)

    tf = getattr(sig, "timeframe", "1m")
    sign = "+" if trade.pnl_points >= 0 else ""
    ax.set_title(
        f"#{trade_no}  {tf} 高檔M頭空  {_fmt_time(sig.timestamp)} → {_fmt_time(trade.exit_time)}  "
        f"{trade.exit_reason}  {sign}{trade.pnl_points:.1f}pt",
        color="#e8f0ea",
        fontsize=11,
    )
    ax.legend(loc="upper left", fontsize=7, frameon=False, labelcolor="#c8d5cc", ncol=7)
    step = max(1, len(window) // 6)
    ticks = list(range(0, len(window), step))
    axv.set_xticks(ticks)
    axv.set_xticklabels(
        [_fmt_time(window.index[i]) for i in ticks],
        color="#8aa193",
        rotation=20,
        ha="right",
    )
    fig.tight_layout(pad=0.45)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def _img_data_uri(path: Path) -> str:
    return f"data:image/png;base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _render_trade_card(
    df: pd.DataFrame,
    trade: TradeResult,
    trade_no: int,
    img_href: str,
) -> str:
    sig = trade.signal
    p = sig.pattern
    pnl_class = "pnl-win" if trade.pnl_points > 0 else ("pnl-flat" if trade.pnl_points == 0 else "pnl-loss")
    tag_class = {
        "take_profit": "tag-tp",
        "stop_loss": "tag-sl",
        "trail_stop": "tag-trail",
    }.get(trade.exit_reason, "tag-time")
    gap = abs(p.first_high - p.second_high)
    avg = (p.first_high + p.second_high) / 2
    gap_pct = gap / avg * 100 if avg else 0
    risk = sig.risk
    tf = getattr(sig, "timeframe", "1m")
    extra_bits = [s for s in (m5_snapshot(df, sig.bar_idx), htf_snapshot(df, sig.bar_idx, "ma60_1h", "1h MA60")) if s]
    extra_ma = ("\n" + "\n".join(extra_bits)) if extra_bits else ""
    return (
        "<article class='trade-card'>"
        "<header class='card-header'>"
        f"<div class='card-title'><span class='trade-no'>#{trade_no}</span>"
        f"<span class='trade-time'>{escape(_fmt_time(sig.timestamp))} → {escape(_fmt_time(trade.exit_time))}</span></div>"
        f"<div class='card-pnl {pnl_class}'>{trade.pnl_points:+.1f} pts</div>"
        "</header>"
        "<div class='tags'>"
        f"<span class='tag {tag_class}'>{escape(trade.exit_reason)}</span>"
        "<span class='tag tag-info'>M頭</span>"
        f"<span class='tag tag-info'>{escape(tf)} 空</span>"
        "</div>"
        "<pre class='trade-detail'>"
        f"entry(跌破MA60+頸線) {sig.entry:.2f}\n"
        f"MA60 {sig.ma60:.2f} / MA20 {sig.ma20:.2f} / MA5 {sig.ma5:.2f}{extra_ma}\n"
        f"均線帶寬 {getattr(sig, 'ribbon_spread', 0.0):.1f}（≥28 才進，糾結濾掉）\n"
        f"stop H高點+緩衝 {sig.stop_loss:.2f}  (風險 {risk:.1f})\n"
        f"TP {sig.target:.2f}\n"
        f"exit {trade.exit_price:.2f}  {trade.exit_reason}\n"
        f"M頭 H1 {p.first_high:.2f} / H2 {p.second_high:.2f}\n"
        f"頸線 {p.neckline:.2f} / 深度 {p.depth:.2f}\n"
        f"雙頂價差 {gap_pct:.2f}%\n"
        f"$ {trade.pnl_dollars:+,.2f} NQ×1"
        "</pre>"
        f"<div class='mini-chart'><img src='{escape(img_href)}' alt='#{trade_no}' "
        "style='width:100%;display:block;border-radius:10px'/></div>"
        "</article>"
    )


def _funnel_html(funnel: Optional[Dict[str, int]]) -> str:
    if not funnel:
        return ""
    return (
        f"<p class='muted'>漏斗：M頭 {funnel.get('m_heads', 0)} → "
        f"高檔 {funnel.get('high_level', 0)} → "
        f"進場 {funnel.get('taken', 0)}"
        f"（非高檔 {funnel.get('skip_not_high', 0)} · "
        f"伸幅不足 {funnel.get('skip_thin_ext', 0)} · "
        f"未破MA60 {funnel.get('skip_no_ma60', 0)} · "
        f"均線糾結 {funnel.get('skip_tangled', 0)} · "
        f"慢均夾心 {funnel.get('skip_sandwich', 0)} · "
        f"MA200太遠 {funnel.get('skip_far_ma200', 0)} · "
        f"1h未測 {funnel.get('skip_htf_gap', 0)} · "
        f"破高失效 {funnel.get('skip_invalidated', 0)} · "
        f"風險過窄 {funnel.get('skip_tiny_risk', 0)} · "
        f"風險過寬 {funnel.get('skip_wide_risk', 0)}）</p>"
    )


def _stats_cards(stats: dict) -> str:
    total_cls = "pnl-win" if stats["total_pnl_points"] >= 0 else "pnl-loss"
    return (
        '<div class="cards">'
        f'<div class="card">筆數<b>{stats["trades"]}</b></div>'
        f'<div class="card">勝率<b>{stats["win_rate"] * 100:.1f}%</b></div>'
        f'<div class="card">總點數<b class="{total_cls}">{stats["total_pnl_points"]:+.1f}</b></div>'
        f'<div class="card">勝/負<b>{stats["wins"]}/{stats["losses"]}</b></div>'
        "</div>"
        f'<p class="muted">均損益 {stats["avg_pnl_points"]:+.1f} 點 · ${stats["total_pnl_dollars"]:+,.0f}（NQ×1）</p>'
    )


def _render_cards(
    df: pd.DataFrame,
    trades: List[TradeResult],
    img_dir: Path,
    *,
    prefix: str,
    embed_images: bool,
) -> str:
    cards: List[str] = []
    for i, trade in enumerate(trades, 1):
        img_name = _trade_img_name(trade, i, prefix=prefix)
        png = draw_trade_png(df, trade, img_dir / img_name, i)
        href = _img_data_uri(png) if embed_images else f"img/{img_name}"
        cards.append(_render_trade_card(df, trade, i, href))
    return "".join(cards) or "<div class='empty'>未偵測到高檔M頭跌破MA60訊號</div>"


def write_html_report(
    path: str | Path,
    df: pd.DataFrame,
    trades: List[TradeResult],
    symbol: str,
    period: str,
    funnel: Optional[Dict[str, int]] = None,
    *,
    embed_images: bool = False,
    note: str = "",
    m5_df: pd.DataFrame | None = None,
    m5_trades: List[TradeResult] | None = None,
    m5_funnel: Optional[Dict[str, int]] = None,
    h1_df: pd.DataFrame | None = None,
    h1_trades: List[TradeResult] | None = None,
    h1_funnel: Optional[Dict[str, int]] = None,
) -> Path:
    stats = summarize(trades)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    img_dir = out.parent / "img"
    cards = _render_cards(df, trades, img_dir, prefix="m", embed_images=embed_images)
    start = df.index[0].strftime("%Y-%m-%d %H:%M")
    end = df.index[-1].strftime("%Y-%m-%d %H:%M")
    note_line = f"<p class='muted'>{escape(note)}</p>" if note else ""

    m5_block = ""
    h1_block = ""
    compare_line = ""
    extra_stats: List[str] = []
    if m5_df is not None and m5_trades is not None:
        m5_stats = summarize(m5_trades)
        extra_stats.append(f"5m {m5_stats['trades']} 筆 {m5_stats['total_pnl_points']:+.1f} 點")
        m5_start = m5_df.index[0].strftime("%Y-%m-%d %H:%M")
        m5_end = m5_df.index[-1].strftime("%Y-%m-%d %H:%M")
        d1 = (df.index[-1] - df.index[0]).total_seconds() / 86400
        d5 = (m5_df.index[-1] - m5_df.index[0]).total_seconds() / 86400
        window_note = ""
        if d5 - d1 >= 7:
            window_note = (
                f" Yahoo 1m 只能回看約 {d1:.0f} 天；五分K 這次是 {d5:.0f} 天。"
            )
        m5_cards = _render_cards(m5_df, m5_trades, img_dir, prefix="f", embed_images=embed_images)
        m5_block = f"""
<section class="summary">
<h1>五分K 對照 · 同一套高檔M頭跌破MA60</h1>
<p class="muted">5m · {escape(m5_start)} → {escape(m5_end)} ET · bars={len(m5_df)}</p>
<p class="muted">轉折確認 3 根（15 分）、雙頂間隔 4–48 根（20 分–4 小時）、近 2 小時高點、2R。帶寬未滿 28 點不進。收盤夾在 MA120/MA200 中間不空。收盤還在 MA200 上方超過 150 點且 1h 已破，不空。已破 MA200 但 1h MA60 還在下面超過 200 點、且 MA60 仍往上，也不空。停損頭頂 +36。0.75R 鎖 0.5R、1.2R 鎖 0.9R、1.6R 鎖 1.2R。鎖利下一根生效。無破底翻平空、無硬虧 +60。</p>
{_stats_cards(m5_stats)}
{_funnel_html(m5_funnel)}
<div class="equity">{_equity_svg([t.pnl_points for t in m5_trades])}</div>
</section>
{m5_cards}
"""
        compare_line = (
            f"<p class='muted'>對照：1m {stats['trades']} 筆 {stats['total_pnl_points']:+.1f} 點 · "
            + " · ".join(extra_stats)
            + f"（規則相同，K 數換成約略同一鐘面時間）{window_note}</p>"
        )
    if h1_df is not None and h1_trades is not None:
        h1_stats = summarize(h1_trades)
        extra_stats.append(f"1h {h1_stats['trades']} 筆 {h1_stats['total_pnl_points']:+.1f} 點")
        h1_start = h1_df.index[0].strftime("%Y-%m-%d %H:%M")
        h1_end = h1_df.index[-1].strftime("%Y-%m-%d %H:%M")
        h1_cards = _render_cards(h1_df, h1_trades, img_dir, prefix="h", embed_images=embed_images)
        h1_block = f"""
<section class="summary">
<h1>一小時K 對照 · 同一套高檔M頭跌破MA60</h1>
<p class="muted">1h · {escape(h1_start)} → {escape(h1_end)} ET · bars={len(h1_df)}</p>
<p class="muted">轉折確認 2 根（2 小時）、雙頂間隔 2–16 根、近 8 小時高點、2R。帶寬未滿 28 點不進。收盤夾在 MA120/MA200 中間不空。停損頭頂 +50。1h MA60 約等於 2.5 天，能破的比較少。</p>
{_stats_cards(h1_stats)}
{_funnel_html(h1_funnel)}
<div class="equity">{_equity_svg([t.pnl_points for t in h1_trades])}</div>
</section>
{h1_cards}
"""
        compare_line = (
            f"<p class='muted'>對照：1m {stats['trades']} 筆 {stats['total_pnl_points']:+.1f} 點 · "
            + " · ".join(extra_stats)
            + "（規則相同，K 數換成約略同一鐘面時間）</p>"
        )

    html = f"""<!DOCTYPE html>
<html lang="zh-Hant"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"/>
<title>{escape(symbol)} 高檔M頭跌破MA60 · 1m / 5m / 1h</title>
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
.tag-trail{{background:rgba(0,200,180,0.14);color:#5eead4;border-color:rgba(0,200,180,0.35)}}
.tag-reclaim{{background:rgba(167,139,250,0.14);color:#c4b5fd;border-color:rgba(167,139,250,0.35)}}
.tag-info{{background:rgba(88,166,255,0.12);color:#79c0ff;border-color:rgba(88,166,255,0.28)}}
.trade-detail{{margin:0 0 10px;padding:10px 12px;background:#0d1117;border-radius:10px;border:1px solid #21262d;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;line-height:1.55;color:#c9d1d9;white-space:pre-wrap}}
.mini-chart{{margin:0 -6px -4px;border-radius:10px;overflow:hidden}}
.empty{{text-align:center;color:#8b949e;padding:40px 16px;background:#161b22;border-radius:14px;border:1px solid #30363d}}
</style></head><body>
<div class="page">
<section class="summary">
<h1>{escape(symbol)} 一分K 高檔M頭 · 跌破MA60做空</h1>
<p class="muted">{escape(period)} · {escape(start)} → {escape(end)} ET · bars={len(df)}</p>
<p class="muted">高檔雙頂確認後，收盤同時跌破頸線與 MA60（≥8 點）。MA5/10/20/30/60 還黏成一團（帶寬 &lt; 28）先等打開。1m 停損頭頂 +8、1.6R 鎖 1.2R；5m 停損 +36，0.75 / 1.2 / 1.6R 鎖利。鎖利下一根才生效。</p>
{note_line}
{compare_line}
{_stats_cards(stats)}
{_funnel_html(funnel)}
<div class="equity">{_equity_svg([t.pnl_points for t in trades])}</div>
</section>
{cards}
{m5_block}
{h1_block}
</div>
</body></html>
"""
    out.write_text(html, encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cmd_backtest(args) -> int:
    extra = {}
    if getattr(args, "rth", False):
        extra["session_start"] = 9 * 60 + 30
        extra["session_end"] = 16 * 60

    df = to_et(load_bars(args.symbol, "1m", args.period))
    if df.empty:
        print("no data", file=sys.stderr)
        return 1
    df5 = to_et(load_bars(args.symbol, "5m", args.period))
    df1h = to_et(load_bars(args.symbol, "1h", args.period))
    if not df5.empty:
        df = overlay_m5_ma60(df, df5)
    if not df1h.empty:
        df = overlay_htf_ma60(df, df1h, col="ma60_1h")
        if not df5.empty:
            df5 = overlay_htf_ma60(df5, df1h, col="ma60_1h")

    _, trades, funnel = run_tf_backtest(df, "1m", extra)
    stats = summarize(trades)
    print(f"{args.symbol} 1m {args.period} bars={len(df)} {df.index[0]} -> {df.index[-1]}")
    print(
        f"1m trades={stats['trades']} WR={stats['win_rate'] * 100:.1f}% "
        f"pnl={stats['total_pnl_points']:+.1f} ${stats['total_pnl_dollars']:+,.0f}"
    )
    print(
        "1m funnel "
        f"m={funnel.get('m_heads', 0)} high={funnel.get('high_level', 0)} "
        f"taken={funnel.get('taken', 0)} "
        f"not_high={funnel.get('skip_not_high', 0)} "
        f"thin={funnel.get('skip_thin_ext', 0)} "
        f"no_ma60={funnel.get('skip_no_ma60', 0)} "
        f"tangled={funnel.get('skip_tangled', 0)} "
        f"invalid={funnel.get('skip_invalidated', 0)} "
        f"tiny={funnel.get('skip_tiny_risk', 0)} "
        f"wide={funnel.get('skip_wide_risk', 0)}"
    )
    for i, t in enumerate(trades, 1):
        snap = m5_snapshot(df, t.signal.bar_idx)
        extra_s = f"  {snap}" if snap else ""
        ribbon = getattr(t.signal, "ribbon_spread", 0.0)
        print(
            f"  [1m {i}] {_fmt_time(t.signal.timestamp)} -> {_fmt_time(t.exit_time)} "
            f"{t.exit_reason} {t.pnl_points:+.1f}  "
            f"entry={t.signal.entry:.2f} stop={t.signal.stop_loss:.2f} tp={t.signal.target:.2f} "
            f"ribbon={ribbon:.1f}"
            f"{extra_s}"
        )

    m5_trades: List[TradeResult] = []
    m5_funnel: Dict[str, int] = {}
    if not df5.empty:
        _, m5_trades, m5_funnel = run_tf_backtest(df5, "5m", extra)
        m5_stats = summarize(m5_trades)
        print(f"{args.symbol} 5m {args.period} bars={len(df5)} {df5.index[0]} -> {df5.index[-1]}")
        print(
            f"5m trades={m5_stats['trades']} WR={m5_stats['win_rate'] * 100:.1f}% "
            f"pnl={m5_stats['total_pnl_points']:+.1f} ${m5_stats['total_pnl_dollars']:+,.0f}"
        )
        print(
            "5m funnel "
            f"m={m5_funnel.get('m_heads', 0)} high={m5_funnel.get('high_level', 0)} "
            f"taken={m5_funnel.get('taken', 0)} "
            f"not_high={m5_funnel.get('skip_not_high', 0)} "
            f"thin={m5_funnel.get('skip_thin_ext', 0)} "
            f"no_ma60={m5_funnel.get('skip_no_ma60', 0)} "
            f"tangled={m5_funnel.get('skip_tangled', 0)} "
            f"sandwich={m5_funnel.get('skip_sandwich', 0)} "
            f"far200={m5_funnel.get('skip_far_ma200', 0)} "
            f"htfgap={m5_funnel.get('skip_htf_gap', 0)} "
            f"invalid={m5_funnel.get('skip_invalidated', 0)}"
        )
        for i, t in enumerate(m5_trades, 1):
            ribbon = getattr(t.signal, "ribbon_spread", 0.0)
            h1s = htf_snapshot(df5, t.signal.bar_idx, "ma60_1h", "1h MA60")
            extra_h = f"  {h1s}" if h1s else ""
            print(
                f"  [5m {i}] {_fmt_time(t.signal.timestamp)} -> {_fmt_time(t.exit_time)} "
                f"{t.exit_reason} {t.pnl_points:+.1f}  "
                f"entry={t.signal.entry:.2f} stop={t.signal.stop_loss:.2f} tp={t.signal.target:.2f} "
                f"ribbon={ribbon:.1f}"
                f"{extra_h}"
            )

    h1_trades: List[TradeResult] = []
    h1_funnel: Dict[str, int] = {}
    if not df1h.empty:
        _, h1_trades, h1_funnel = run_tf_backtest(df1h, "1h", extra)
        h1_stats = summarize(h1_trades)
        print(f"{args.symbol} 1h {args.period} bars={len(df1h)} {df1h.index[0]} -> {df1h.index[-1]}")
        print(
            f"1h trades={h1_stats['trades']} WR={h1_stats['win_rate'] * 100:.1f}% "
            f"pnl={h1_stats['total_pnl_points']:+.1f} ${h1_stats['total_pnl_dollars']:+,.0f}"
        )
        print(
            "1h funnel "
            f"m={h1_funnel.get('m_heads', 0)} high={h1_funnel.get('high_level', 0)} "
            f"taken={h1_funnel.get('taken', 0)} "
            f"not_high={h1_funnel.get('skip_not_high', 0)} "
            f"no_ma60={h1_funnel.get('skip_no_ma60', 0)} "
            f"sandwich={h1_funnel.get('skip_sandwich', 0)} "
            f"invalid={h1_funnel.get('skip_invalidated', 0)}"
        )
        for i, t in enumerate(h1_trades, 1):
            print(
                f"  [1h {i}] {_fmt_time(t.signal.timestamp)} -> {_fmt_time(t.exit_time)} "
                f"{t.exit_reason} {t.pnl_points:+.1f}  "
                f"entry={t.signal.entry:.2f} stop={t.signal.stop_loss:.2f} tp={t.signal.target:.2f}"
            )

    html_path = args.html
    if args.pages:
        html_path = html_path or str(PAGES_HTML)
    if html_path:
        kw = dict(
            funnel=funnel,
            m5_df=df5 if not df5.empty else None,
            m5_trades=m5_trades if not df5.empty else None,
            m5_funnel=m5_funnel if not df5.empty else None,
            h1_df=df1h if not df1h.empty else None,
            h1_trades=h1_trades if not df1h.empty else None,
            h1_funnel=h1_funnel if not df1h.empty else None,
        )
        out = write_html_report(html_path, df, trades, args.symbol, args.period, **kw)
        view = Path(html_path).with_name("view.html")
        if args.pages or Path(html_path).name == "index.html":
            write_html_report(
                view,
                df,
                trades,
                args.symbol,
                args.period,
                embed_images=True,
                note="圖已內嵌，手機請往下捲。粉紅虛線是 5m MA60，紫虛線是 1h MA60。",
                **kw,
            )
            print(f"view={view}")
        print(f"html={out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NQ 一分K 高檔M頭，收盤跌破MA60做空")
    p.add_argument("--symbol", default="NQ=F")
    p.add_argument("--period", default="30d")
    p.add_argument("--html", default="")
    p.add_argument("--pages", action="store_true", help="寫到 docs/nq-m-head/index.html")
    p.add_argument("--rth", action="store_true", help="只做 09:30–16:00 ET")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return cmd_backtest(args)


if __name__ == "__main__":
    raise SystemExit(main())

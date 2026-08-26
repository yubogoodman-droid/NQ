"""NQ 五分 K：破底 → 站上 MA10 → 回踩 MA10 做多。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class Signal:
    break_idx: int
    reclaim_idx: int
    entry_idx: int
    entry_price: float
    stop_price: float
    target_price: float
    break_low: float
    prior_low: float
    ma5: float
    ma10: float
    ma20: float
    ma30: float
    ma60: float
    quality: str = "C"
    quality_score: int = 0
    pierce: float = 0.0
    break_depth: float = 0.0


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


def quality_from_setup(break_depth: float, pierce: float, ma5_above_ma10: bool) -> Tuple[int, str]:
    score = 0
    if break_depth >= 40.0:
        score += 1
    if pierce <= 8.0:
        score += 1
    if ma5_above_ma10:
        score += 1
    if score >= 2:
        return score, "A"
    if score == 1:
        return score, "B"
    return score, "C"


def _col(df: pd.DataFrame, name: str) -> str:
    lower = {c.lower(): c for c in df.columns}
    key = name.lower()
    if key not in lower:
        raise ValueError(f"DataFrame 缺少欄位: {name}")
    return lower[key]


def detect_signals(
    df: pd.DataFrame,
    *,
    lookback_bars: int = 24,
    min_break_depth: float = 20.0,
    reclaim_window: int = 18,
    pullback_window: int = 18,
    min_hold_above: int = 1,
    max_pierce: float = 25.0,
    stop_buffer: float = 12.0,
    target_r: float = 2.0,
    max_risk: float = 80.0,
    min_risk: float = 8.0,
    use_structural_stop: bool = False,
    skip_hour_start: Optional[int] = None,
    skip_hour_end: Optional[int] = None,
    min_entry_gap: int = 12,
    funnel: Optional[Dict[str, int]] = None,
) -> List[Signal]:
    """
    五分 K 做多三步：

    1. 破底：跌破近 lookback 根低點，深度 ≥ min_break_depth，且低點在 MA10 下方
    2. 站上 MA10：破底後 reclaim_window 根內收盤站上 MA10
    3. 回踩 MA10：至少 min_hold_above 根低點完全在 MA10 上方後，低點回踩均線、收盤仍站上
    """
    close = df[_col(df, "close")].to_numpy(float)
    low = df[_col(df, "low")].to_numpy(float)

    ma5 = sma(close, 5)
    ma10 = sma(close, 10)
    ma20 = sma(close, 20)
    ma30 = sma(close, 30)
    ma60 = sma(close, 60)
    prior_low = rolling_min_prev(low, lookback_bars)

    n = len(close)
    signals: List[Signal] = []
    last_entry = -(10**9)
    warmup = max(lookback_bars, 60)
    i = warmup
    fun = funnel if funnel is not None else {}

    def bump(key: str) -> None:
        fun[key] = fun.get(key, 0) + 1

    while i < n - 1:
        if np.isnan(prior_low[i]) or np.isnan(ma10[i]):
            i += 1
            continue

        support = float(prior_low[i])
        if low[i] >= support:
            i += 1
            continue

        depth = support - float(low[i])
        if depth < min_break_depth:
            bump("shallow")
            i += 1
            continue
        if float(low[i]) >= float(ma10[i]):
            bump("break_still_above_ma10")
            i += 1
            continue

        bump("break")
        break_idx = i
        break_low = float(low[i])
        reclaimed = False
        reclaim_idx: Optional[int] = None
        bars_above = 0
        entered = False
        scan_end = min(break_idx + reclaim_window + pullback_window + 1, n)

        for j in range(break_idx, scan_end):
            if np.isnan(ma10[j]):
                continue

            if float(low[j]) < break_low:
                break_low = float(low[j])
                break_idx = j
                reclaimed = False
                reclaim_idx = None
                bars_above = 0

            if not reclaimed:
                if j > break_idx + reclaim_window:
                    bump("skip_no_reclaim")
                    break
                if close[j] > ma10[j]:
                    bump("reclaim")
                    reclaimed = True
                    reclaim_idx = j
                    bars_above = 1 if float(low[j]) > float(ma10[j]) else 0
                continue

            assert reclaim_idx is not None
            if j - reclaim_idx > pullback_window:
                bump("skip_no_pullback")
                break
            if close[j] < ma10[j]:
                bump("skip_lost_stand")
                break

            if float(low[j]) > float(ma10[j]):
                bars_above += 1
                continue

            if j == reclaim_idx:
                continue

            pierce = float(ma10[j]) - float(low[j])
            if pierce > max_pierce:
                bump("skip_deep_pierce")
                break
            if bars_above < min_hold_above:
                continue
            if skip_hour_start is not None and skip_hour_end is not None:
                hour = df.index[j].hour
                if skip_hour_start <= hour < skip_hour_end:
                    bump("skip_open_hour")
                    continue
            if j - last_entry < min_entry_gap:
                bump("skip_entry_gap")
                break

            entry = float(close[j])
            if use_structural_stop:
                stop = break_low - stop_buffer
            else:
                stop = min(float(low[j]), float(ma10[j])) - stop_buffer
            risk = entry - stop
            if risk < min_risk:
                bump("skip_tiny_risk")
                continue
            if max_risk > 0 and risk > max_risk:
                bump("skip_max_risk")
                continue

            target = entry + risk * target_r
            ma5_v = float(ma5[j]) if not np.isnan(ma5[j]) else entry
            ma10_v = float(ma10[j])
            q_score, q_grade = quality_from_setup(support - break_low, pierce, ma5_v > ma10_v)
            bump("taken")
            signals.append(
                Signal(
                    break_idx=break_idx,
                    reclaim_idx=reclaim_idx,
                    entry_idx=j,
                    entry_price=entry,
                    stop_price=stop,
                    target_price=target,
                    break_low=break_low,
                    prior_low=support,
                    ma5=ma5_v,
                    ma10=ma10_v,
                    ma20=float(ma20[j]) if not np.isnan(ma20[j]) else 0.0,
                    ma30=float(ma30[j]) if not np.isnan(ma30[j]) else 0.0,
                    ma60=float(ma60[j]) if not np.isnan(ma60[j]) else 0.0,
                    quality=q_grade,
                    quality_score=q_score,
                    pierce=pierce,
                    break_depth=support - break_low,
                )
            )
            last_entry = j
            entered = True
            i = j + 4
            break

        if not entered:
            i = break_idx + 1

    return signals


def simulate(
    df: pd.DataFrame,
    signals: Sequence[Signal],
    *,
    max_hold: int = 24,
) -> List[TradeResult]:
    close = df[_col(df, "close")].to_numpy(float)
    high = df[_col(df, "high")].to_numpy(float)
    low = df[_col(df, "low")].to_numpy(float)
    n = len(close)
    results: List[TradeResult] = []
    busy_until = -1

    for sig in signals:
        e = sig.entry_idx
        if e <= busy_until:
            continue
        entry = sig.entry_price
        stop = sig.stop_price
        target = sig.target_price
        exit_idx = min(e + max_hold, n - 1)
        exit_price = float(close[exit_idx])
        reason = "time"

        for k in range(e + 1, min(e + max_hold, n - 1) + 1):
            if float(low[k]) <= stop:
                exit_idx, exit_price, reason = k, stop, "stop"
                break
            if float(high[k]) >= target:
                exit_idx, exit_price, reason = k, target, "target"
                break

        busy_until = exit_idx
        results.append(
            TradeResult(
                signal=sig,
                entry_idx=e,
                exit_idx=exit_idx,
                entry_price=entry,
                exit_price=exit_price,
                stop_price=stop,
                target_price=target,
                pnl_points=float(exit_price - entry),
                exit_reason=reason,
                quality=sig.quality,
            )
        )
    return results


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
        "avg": float(sum(pnls) / n) if n else 0.0,
        "by_quality": {
            q: {
                "n": len(v),
                "wins": sum(1 for p in v if p > 0),
                "pnl": float(sum(v)),
            }
            for q, v in sorted(by_q.items())
        },
    }

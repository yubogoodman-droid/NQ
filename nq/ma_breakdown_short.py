"""NQ 五分 K：收盤同時跌破 MA5/10/20/30/60/120 做空。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

MA_PERIODS = (5, 10, 20, 30, 60, 120)


@dataclass
class Signal:
    entry_idx: int
    entry_price: float
    stop_price: float
    target_price: float
    ma5: float
    ma10: float
    ma20: float
    ma30: float
    ma60: float
    ma120: float
    stack_high: float
    stack_low: float
    quality: str = "C"
    quality_score: int = 0
    break_span: float = 0.0


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


def _col(df: pd.DataFrame, name: str) -> str:
    lower = {c.lower(): c for c in df.columns}
    key = name.lower()
    if key not in lower:
        raise ValueError(f"DataFrame 缺少欄位: {name}")
    return lower[key]


def _below_all(close: float, mas: Sequence[float]) -> bool:
    return all(close < m for m in mas)


def _above_all(close: float, mas: Sequence[float]) -> bool:
    return all(close >= m for m in mas)


def quality_from_break(break_span: float, dist_ma120: float, bearish: bool) -> Tuple[int, str]:
    score = 0
    if break_span >= 40.0:
        score += 1
    if dist_ma120 >= 20.0:
        score += 1
    if bearish:
        score += 1
    if score >= 2:
        return score, "A"
    if score == 1:
        return score, "B"
    return score, "C"


def detect_signals(
    df: pd.DataFrame,
    *,
    stop_buffer: float = 12.0,
    target_r: float = 2.0,
    max_risk: float = 120.0,
    min_risk: float = 8.0,
    min_break_pts: float = 20.0,
    require_bearish: bool = True,
    session_start: Optional[int] = 8,
    session_end: Optional[int] = 16,
    min_entry_gap: int = 12,
    funnel: Optional[Dict[str, int]] = None,
) -> List[Signal]:
    """
    五分 K 做空：

    上一根收盤仍全部在 MA5/10/20/30/60/120 之上（含貼均），
    本根收盤才一次全部低於這六條。已先掉在部分均線下、再補破其餘的不算。
    """
    close = df[_col(df, "close")].to_numpy(float)
    open_ = df[_col(df, "open")].to_numpy(float)
    high = df[_col(df, "high")].to_numpy(float)

    mas = {n: sma(close, n) for n in MA_PERIODS}
    n = len(close)
    signals: List[Signal] = []
    last_entry = -(10**9)
    warmup = max(MA_PERIODS)
    fun = funnel if funnel is not None else {}

    def bump(key: str) -> None:
        fun[key] = fun.get(key, 0) + 1

    def stack_at(i: int) -> Optional[List[float]]:
        vals = [float(mas[p][i]) for p in MA_PERIODS]
        if any(np.isnan(v) for v in vals):
            return None
        return vals

    for i in range(warmup, n - 1):
        cur = stack_at(i)
        prev = stack_at(i - 1)
        if cur is None or prev is None:
            continue

        now_below = _below_all(float(close[i]), cur)
        was_above = _above_all(float(close[i - 1]), prev)
        if not now_below:
            continue
        bump("below_all")
        if not was_above:
            bump("not_simultaneous")
            continue
        bump("fresh_break")

        if max(cur) - float(close[i]) < min_break_pts:
            bump("skip_thin")
            continue

        if require_bearish and float(close[i]) >= float(open_[i]):
            bump("skip_not_bear")
            continue

        hour = df.index[i].hour
        if session_start is not None and hour < session_start:
            bump("skip_session")
            continue
        if session_end is not None and hour >= session_end:
            bump("skip_session")
            continue
        if i - last_entry < min_entry_gap:
            bump("skip_entry_gap")
            continue

        entry = float(close[i])
        stop = float(high[i]) + stop_buffer
        risk = stop - entry
        if risk < min_risk:
            bump("skip_tiny_risk")
            continue
        if max_risk > 0 and risk > max_risk:
            bump("skip_max_risk")
            continue

        stack_high = max(cur)
        stack_low = min(cur)
        break_span = stack_high - stack_low
        q_score, q_grade = quality_from_break(
            break_span, float(cur[-1] - entry), float(close[i]) < float(open_[i])
        )
        bump("taken")
        signals.append(
            Signal(
                entry_idx=i,
                entry_price=entry,
                stop_price=stop,
                target_price=entry - risk * target_r,
                ma5=cur[0],
                ma10=cur[1],
                ma20=cur[2],
                ma30=cur[3],
                ma60=cur[4],
                ma120=cur[5],
                stack_high=stack_high,
                stack_low=stack_low,
                quality=q_grade,
                quality_score=q_score,
                break_span=break_span,
            )
        )
        last_entry = i

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
            if float(high[k]) >= stop:
                exit_idx, exit_price, reason = k, stop, "stop"
                break
            if float(low[k]) <= target:
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
                pnl_points=float(entry - exit_price),
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

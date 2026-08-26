"""破底 W 底型態偵測：L1 左、L2 破底、L3 右。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd


@dataclass(frozen=True)
class WBottomPattern:
    """L1 左腳、L2 中間破底、L3 右腳。"""

    l1_idx: int
    l2_idx: int
    l3_idx: int
    neckline_idx: int
    l1: float
    l2: float
    l3: float
    neckline: float
    reclaim_idx: int
    breakout_idx: int | None = None

    @property
    def first_low_idx(self) -> int:
        return self.l1_idx

    @property
    def first_low(self) -> float:
        return self.l1

    @property
    def second_low_idx(self) -> int:
        return self.l3_idx

    @property
    def second_low(self) -> float:
        return self.l3

    @property
    def spring_idx(self) -> int:
        return self.l2_idx

    @property
    def spring_low(self) -> float:
        return self.l2

    @property
    def stop_loss(self) -> float:
        """停損設於右腳 L3。"""
        return self.l3

    @property
    def target(self) -> float:
        """量度目標：頸線 + (頸線 − L2 破底)。"""
        return self.neckline + (self.neckline - self.l2)


def _is_swing_low(lows: Sequence[float], idx: int, lookback: int) -> bool:
    if idx < lookback or idx >= len(lows) - lookback:
        return False
    pivot = lows[idx]
    window = lows[idx - lookback : idx + lookback + 1]
    return pivot == min(window) and window.count(pivot) == 1


def _find_swing_lows(lows: Sequence[float], lookback: int) -> list[int]:
    return [i for i in range(len(lows)) if _is_swing_low(lows, i, lookback)]


def _is_l3_trough(
    lows: Sequence[float],
    idx: int,
    *,
    l2_idx: int,
    reclaim_idx: int,
    lookback: int,
) -> bool:
    """L3 左側從破底下一根起算，避免 L2 長影線把右腳否決掉。"""
    n = len(lows)
    left_start = max(l2_idx + 1, reclaim_idx)
    if idx <= left_start or idx + lookback >= n:
        return False
    pivot = lows[idx]
    if pivot != min(lows[idx : idx + lookback + 1]):
        return False
    left0 = max(left_start, idx - lookback)
    return pivot == min(lows[left0 : idx + 1])


def detect_w_bottoms(
    df: pd.DataFrame,
    *,
    swing_lookback: int = 3,
    low_tolerance_pct: float = 0.001,
    min_bars_between_lows: int = 8,
    max_bars_between_lows: int = 80,
    min_spring_pct: float = 0.0006,
    min_spring_points: float = 10.0,
    min_bounce_pct: float = 0.001,
    max_reclaim_bars: int = 12,
    max_breakout_bars: int = 36,
    require_neckline_break: bool = True,
) -> list[WBottomPattern]:
    """
    偵測三點破底 W 底。

    結構::

            頸線
           /    \\
          /      \\      / 突破
        L1        \\    L3
                   \\  /
                    \\/ L2 破底
    """
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame 缺少欄位: {missing}")

    lows = df["low"].tolist()
    highs = df["high"].tolist()
    closes = df["close"].tolist()
    n = len(df)

    swing_lows = _find_swing_lows(lows, swing_lookback)
    patterns: list[WBottomPattern] = []

    for l1_idx in swing_lows:
        l1 = lows[l1_idx]
        if l1 <= 0:
            continue

        min_spring = max(l1 * min_spring_pct, min_spring_points)
        min_bounce = l1 * min_bounce_pct
        search_end = min(l1_idx + max_bars_between_lows, n - 1)

        bounce_high = float("-inf")
        bounced = False
        l2_idx: int | None = None
        l2 = float("inf")
        reclaim_idx: int | None = None

        for j in range(l1_idx + 1, search_end + 1):
            if l2_idx is None:
                if highs[j] > bounce_high:
                    bounce_high = highs[j]
                if bounce_high - l1 >= min_bounce:
                    bounced = True
                if bounced and l1 - lows[j] >= min_spring:
                    l2_idx = j
                    l2 = lows[j]
                    if closes[j] > l1:
                        reclaim_idx = j
                continue

            if reclaim_idx is None:
                if lows[j] < l2:
                    l2_idx = j
                    l2 = lows[j]
                if j - l2_idx > max_reclaim_bars:
                    break
                if closes[j] > l1:
                    reclaim_idx = j
                continue

            if l2_idx is None or reclaim_idx is None:
                break
            if j + swing_lookback >= n:
                break
            if not _is_l3_trough(
                lows, j, l2_idx=l2_idx, reclaim_idx=reclaim_idx, lookback=swing_lookback
            ):
                continue
            if j - l1_idx < min_bars_between_lows:
                continue

            l3 = lows[j]
            if l3 <= l2:
                continue
            avg = (l1 + l3) / 2
            if avg == 0 or abs(l1 - l3) / avg > low_tolerance_pct:
                continue

            neck_slice = highs[l1_idx + 1 : j]
            if not neck_slice:
                continue
            neckline = max(neck_slice)
            neckline_idx = l1_idx + 1 + neck_slice.index(neckline)
            if neckline - max(l1, l3) < min_bounce:
                continue

            breakout_idx: int | None = None
            if require_neckline_break:
                start_k = j + swing_lookback
                end_k = min(j + max_breakout_bars, n - 1)
                for k in range(start_k, end_k + 1):
                    if closes[k] > neckline:
                        breakout_idx = k
                        break
                if breakout_idx is None:
                    continue

            patterns.append(
                WBottomPattern(
                    l1_idx=l1_idx,
                    l2_idx=l2_idx,
                    l3_idx=j,
                    neckline_idx=neckline_idx,
                    l1=l1,
                    l2=l2,
                    l3=l3,
                    neckline=neckline,
                    reclaim_idx=reclaim_idx,
                    breakout_idx=breakout_idx,
                )
            )
            break

    return _dedupe_patterns(patterns)


def _dedupe_patterns(patterns: Iterable[WBottomPattern]) -> list[WBottomPattern]:
    by_breakout: dict[int, WBottomPattern] = {}
    for p in patterns:
        if p.breakout_idx is None:
            continue
        depth = p.neckline - p.l2
        existing = by_breakout.get(p.breakout_idx)
        if existing is None or depth > existing.neckline - existing.l2:
            by_breakout[p.breakout_idx] = p
    return sorted(by_breakout.values(), key=lambda p: p.breakout_idx or 0)

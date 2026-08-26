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


def _is_right_trough(lows: Sequence[float], idx: int, lookback: int) -> bool:
    n = len(lows)
    if idx + lookback >= n:
        return False
    return lows[idx] == min(lows[idx : idx + lookback + 1])


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
    max_reclaim_bars: int = 36,
    max_breakout_bars: int = 36,
    require_neckline_break: bool = True,
) -> list[WBottomPattern]:
    """
    偵測三點破底 W 底。

    L2 取 L1 與 L3 之間的最低點，避免把後來略破 L1 的回測當成破底。
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
        search_end = min(l1_idx + max_bars_between_lows, n - 1 - swing_lookback)

        for l3_idx in range(l1_idx + min_bars_between_lows, search_end + 1):
            if not _is_right_trough(lows, l3_idx, swing_lookback):
                continue

            l3 = lows[l3_idx]
            avg = (l1 + l3) / 2
            if avg == 0 or abs(l1 - l3) / avg > low_tolerance_pct:
                continue

            mid_lows = lows[l1_idx + 1 : l3_idx]
            if len(mid_lows) < 3:
                continue
            l2 = min(mid_lows)
            l2_idx = l1_idx + 1 + mid_lows.index(l2)
            if l1 - l2 < min_spring or l3 <= l2:
                continue
            if l2_idx - l1_idx < 2 or l3_idx - l2_idx < 2:
                continue

            left0 = max(l2_idx + 1, l3_idx - swing_lookback)
            if lows[l3_idx] != min(lows[left0 : l3_idx + 1]):
                continue

            neck_slice = highs[l1_idx + 1 : l3_idx]
            neckline = max(neck_slice)
            neckline_idx = l1_idx + 1 + neck_slice.index(neckline)
            if neckline - max(l1, l3) < min_bounce:
                continue

            reclaim_end = min(l3_idx, l2_idx + max_reclaim_bars)
            reclaim_idx: int | None = None
            for k in range(l2_idx, reclaim_end + 1):
                if closes[k] > l1:
                    reclaim_idx = k
                    break
            if reclaim_idx is None or l3_idx <= reclaim_idx:
                continue

            breakout_idx: int | None = None
            if require_neckline_break:
                start_k = l3_idx + swing_lookback
                end_k = min(l3_idx + max_breakout_bars, n - 1)
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
                    l3_idx=l3_idx,
                    neckline_idx=neckline_idx,
                    l1=l1,
                    l2=l2,
                    l3=l3,
                    neckline=neckline,
                    reclaim_idx=reclaim_idx,
                    breakout_idx=breakout_idx,
                )
            )

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

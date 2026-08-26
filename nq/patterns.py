"""破底 W 底型態偵測：兩腳 W，右腳跌破左腳。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd


@dataclass(frozen=True)
class WBottomPattern:
    """已確認的破底 W 底（L2 低於 L1）。"""

    first_low_idx: int
    second_low_idx: int
    neckline_idx: int
    first_low: float
    second_low: float
    neckline: float
    reclaim_idx: int
    breakout_idx: int | None = None

    @property
    def spring_idx(self) -> int:
        """破底就是右腳 L2。"""
        return self.second_low_idx

    @property
    def spring_low(self) -> float:
        return self.second_low

    @property
    def stop_loss(self) -> float:
        """停損設於破底（L2）。"""
        return self.second_low

    @property
    def target(self) -> float:
        """量度目標：頸線 + (頸線 − L2)。"""
        depth = self.neckline - self.second_low
        return self.neckline + depth


def _is_swing_low(lows: Sequence[float], idx: int, lookback: int) -> bool:
    if idx < lookback or idx >= len(lows) - lookback:
        return False
    pivot = lows[idx]
    window = lows[idx - lookback : idx + lookback + 1]
    return pivot == min(window) and window.count(pivot) == 1


def _find_swing_lows(lows: Sequence[float], lookback: int) -> list[int]:
    return [i for i in range(len(lows)) if _is_swing_low(lows, i, lookback)]


def detect_w_bottoms(
    df: pd.DataFrame,
    *,
    swing_lookback: int = 3,
    min_bars_between_lows: int = 5,
    max_bars_between_lows: int = 60,
    min_spring_pct: float = 0.0004,
    min_spring_points: float = 8.0,
    max_spring_pct: float = 0.004,
    min_bounce_pct: float = 0.001,
    max_reclaim_bars: int = 12,
    max_breakout_bars: int = 36,
    require_neckline_break: bool = True,
    low_tolerance_pct: float | None = None,
) -> list[WBottomPattern]:
    """
    偵測兩腳破底 W 底。

    結構::

            頸線
           /    \\
          /      \\      / 突破
        L1        \\    /
                   \\  /
                    \\/ L2 破底（右腳跌破 L1 後收復）

    條件：
    1. L1、L2 皆為波段低點
    2. 兩低點之間有頸線（最高點）
    3. L2 明顯低於 L1（破底），但不是崩跌
    4. L2 之後在 max_reclaim_bars 內收盤站回 L1
    5. 收盤突破頸線
    """
    del low_tolerance_pct  # 舊參數：此型態 L2 必須低於 L1，不再要求雙底等高

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

    for i, first_idx in enumerate(swing_lows):
        first_low = lows[first_idx]
        if first_low <= 0:
            continue
        min_spring = max(first_low * min_spring_pct, min_spring_points)
        max_spring = first_low * max_spring_pct
        min_bounce = first_low * min_bounce_pct

        for second_idx in swing_lows[i + 1 :]:
            gap = second_idx - first_idx
            if gap < min_bars_between_lows:
                continue
            if gap > max_bars_between_lows:
                break

            second_low = lows[second_idx]
            undercut = first_low - second_low
            if undercut < min_spring or undercut > max_spring:
                continue

            neck_slice = highs[first_idx + 1 : second_idx]
            if not neck_slice:
                continue
            neckline_price = max(neck_slice)
            neckline_idx = first_idx + 1 + neck_slice.index(neckline_price)
            if neckline_price - first_low < min_bounce:
                continue

            reclaim_idx: int | None = None
            reclaim_end = min(second_idx + max_reclaim_bars, n - 1)
            for k in range(second_idx, reclaim_end + 1):
                if closes[k] > first_low:
                    reclaim_idx = k
                    break
            if reclaim_idx is None:
                continue

            breakout_idx: int | None = None
            if require_neckline_break:
                start_k = max(second_idx + swing_lookback, reclaim_idx)
                end_k = min(second_idx + max_breakout_bars, n - 1)
                for k in range(start_k, end_k + 1):
                    if closes[k] > neckline_price:
                        breakout_idx = k
                        break
                if breakout_idx is None:
                    continue

            patterns.append(
                WBottomPattern(
                    first_low_idx=first_idx,
                    second_low_idx=second_idx,
                    neckline_idx=neckline_idx,
                    first_low=first_low,
                    second_low=second_low,
                    neckline=neckline_price,
                    reclaim_idx=reclaim_idx,
                    breakout_idx=breakout_idx,
                )
            )

    return _dedupe_patterns(patterns)


def _dedupe_patterns(patterns: Iterable[WBottomPattern]) -> list[WBottomPattern]:
    """同一突破 K 只保留破底深度與風險報酬較佳的型態。"""
    by_breakout: dict[int, WBottomPattern] = {}
    for p in patterns:
        if p.breakout_idx is None:
            continue
        depth = p.neckline - p.second_low
        existing = by_breakout.get(p.breakout_idx)
        if existing is None or depth > existing.neckline - existing.second_low:
            by_breakout[p.breakout_idx] = p
    return sorted(by_breakout.values(), key=lambda p: p.breakout_idx or 0)

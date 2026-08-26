"""破底 W 底（中間掃停）型態偵測。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd


@dataclass(frozen=True)
class WBottomPattern:
    """已確認的破底 W 底。"""

    first_low_idx: int
    second_low_idx: int
    neckline_idx: int
    first_low: float
    second_low: float
    neckline: float
    spring_idx: int
    spring_low: float
    reclaim_idx: int
    breakout_idx: int | None = None

    @property
    def stop_loss(self) -> float:
        """停損設於第二個低點（收復後的右底）。"""
        return self.second_low

    @property
    def target(self) -> float:
        """量度目標：頸線 + (頸線 − 破底低點)。"""
        depth = self.neckline - self.spring_low
        return self.neckline + depth


def _is_swing_low(lows: Sequence[float], idx: int, lookback: int) -> bool:
    if idx < lookback or idx >= len(lows) - lookback:
        return False
    pivot = lows[idx]
    window = lows[idx - lookback : idx + lookback + 1]
    return pivot == min(window) and window.count(pivot) == 1


def _find_swing_lows(lows: Sequence[float], lookback: int) -> list[int]:
    return [i for i in range(len(lows)) if _is_swing_low(lows, i, lookback)]


def _is_l2_trough(
    lows: Sequence[float],
    idx: int,
    *,
    spring_idx: int,
    reclaim_idx: int,
    lookback: int,
) -> bool:
    """
    L2 不使用對稱 pivot：左側從破底下一根起算，
    避免中間掃停長影線把右底否決掉。
    """
    n = len(lows)
    left_start = max(spring_idx + 1, reclaim_idx)
    if idx <= left_start or idx + lookback >= n:
        return False
    pivot = lows[idx]
    right = lows[idx : idx + lookback + 1]
    if pivot != min(right):
        return False
    left0 = max(left_start, idx - lookback)
    left = lows[left0 : idx + 1]
    return pivot == min(left)


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
    偵測「中間有破底」的 W 底。

    結構::

            頸線
           /    \\
          /      \\      / 突破
        L1        \\    L2
                   \\  /
                    \\/ 破底（掃停後收復）

    條件：
    1. L1 為波段低點
    2. L1 之後先反彈（高點距離 L1 至少 min_bounce_pct）
    3. 反彈後出現破底：low 明顯低於 L1（掃停）
    4. 破底後在 max_reclaim_bars 內收盤站回 L1
    5. 收復後出現 L2：與 L1 價差 ≤ low_tolerance_pct，且高於破底
    6. 收盤突破 L1～L2 之間的頸線
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

    for first_idx in swing_lows:
        first_low = lows[first_idx]
        if first_low <= 0:
            continue

        min_spring = max(first_low * min_spring_pct, min_spring_points)
        min_bounce = first_low * min_bounce_pct
        search_end = min(first_idx + max_bars_between_lows, n - 1)

        bounce_high = float("-inf")
        bounce_idx = first_idx
        bounced = False
        spring_idx: int | None = None
        spring_low = float("inf")
        reclaim_idx: int | None = None

        for j in range(first_idx + 1, search_end + 1):
            if spring_idx is None:
                if highs[j] > bounce_high:
                    bounce_high = highs[j]
                    bounce_idx = j
                if bounce_high - first_low >= min_bounce:
                    bounced = True
                if bounced and first_low - lows[j] >= min_spring:
                    spring_idx = j
                    spring_low = lows[j]
                    if closes[j] > first_low:
                        reclaim_idx = j
                continue

            if reclaim_idx is None:
                if lows[j] < spring_low:
                    spring_idx = j
                    spring_low = lows[j]
                if j - spring_idx > max_reclaim_bars:
                    break
                if closes[j] > first_low:
                    reclaim_idx = j
                continue

            if spring_idx is None or reclaim_idx is None:
                break
            if j + swing_lookback >= n:
                break
            if not _is_l2_trough(
                lows,
                j,
                spring_idx=spring_idx,
                reclaim_idx=reclaim_idx,
                lookback=swing_lookback,
            ):
                continue
            if j - first_idx < min_bars_between_lows:
                continue

            second_low = lows[j]
            if second_low <= spring_low:
                continue
            avg_low = (first_low + second_low) / 2
            if avg_low == 0:
                continue
            if abs(first_low - second_low) / avg_low > low_tolerance_pct:
                continue

            neck_slice = highs[first_idx + 1 : j]
            if not neck_slice:
                continue
            neckline_price = max(neck_slice)
            neckline_idx = first_idx + 1 + neck_slice.index(neckline_price)
            if neckline_price - max(first_low, second_low) < min_bounce:
                continue

            breakout_idx: int | None = None
            if require_neckline_break:
                start_k = j + swing_lookback
                end_k = min(j + max_breakout_bars, n - 1)
                for k in range(start_k, end_k + 1):
                    if closes[k] > neckline_price:
                        breakout_idx = k
                        break
                if breakout_idx is None:
                    continue
            else:
                breakout_idx = None

            patterns.append(
                WBottomPattern(
                    first_low_idx=first_idx,
                    second_low_idx=j,
                    neckline_idx=neckline_idx,
                    first_low=first_low,
                    second_low=second_low,
                    neckline=neckline_price,
                    spring_idx=spring_idx,
                    spring_low=spring_low,
                    reclaim_idx=reclaim_idx,
                    breakout_idx=breakout_idx,
                )
            )
            break

        _ = bounce_idx  # 反彈高點已反映在頸線搜尋中

    return _dedupe_patterns(patterns)


def _dedupe_patterns(patterns: Iterable[WBottomPattern]) -> list[WBottomPattern]:
    """同一突破 K 只保留破底最深、風險報酬較佳的型態。"""
    by_breakout: dict[int, WBottomPattern] = {}
    for p in patterns:
        if p.breakout_idx is None:
            continue
        depth = p.neckline - p.spring_low
        risk = p.neckline - p.second_low
        rr = depth / risk if risk > 0 else 0
        existing = by_breakout.get(p.breakout_idx)
        if existing is None:
            by_breakout[p.breakout_idx] = p
            continue
        ex_depth = existing.neckline - existing.spring_low
        ex_risk = existing.neckline - existing.second_low
        ex_rr = ex_depth / ex_risk if ex_risk > 0 else 0
        if (rr, depth) > (ex_rr, ex_depth):
            by_breakout[p.breakout_idx] = p
    return sorted(by_breakout.values(), key=lambda p: p.breakout_idx or 0)

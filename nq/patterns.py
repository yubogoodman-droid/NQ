"""W 底（雙底）型態偵測。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd


@dataclass(frozen=True)
class WBottomPattern:
    """已確認的 W 底型態。"""

    first_low_idx: int
    second_low_idx: int
    neckline_idx: int
    first_low: float
    second_low: float
    neckline: float
    breakout_idx: int | None = None

    @property
    def stop_loss(self) -> float:
        """停損設於第二個低點下方。"""
        return self.second_low

    @property
    def target(self) -> float:
        """量度目標：頸線 + (頸線 - 最低點)。"""
        depth = self.neckline - min(self.first_low, self.second_low)
        return self.neckline + depth


def _is_swing_low(lows: Sequence[float], idx: int, lookback: int) -> bool:
    if idx < lookback or idx >= len(lows) - lookback:
        return False
    pivot = lows[idx]
    window = lows[idx - lookback : idx + lookback + 1]
    return pivot == min(window) and window.count(pivot) == 1


def _is_swing_high(highs: Sequence[float], idx: int, lookback: int) -> bool:
    if idx < lookback or idx >= len(highs) - lookback:
        return False
    pivot = highs[idx]
    window = highs[idx - lookback : idx + lookback + 1]
    return pivot == max(window) and window.count(pivot) == 1


def _find_swing_lows(lows: Sequence[float], lookback: int) -> list[int]:
    return [i for i in range(len(lows)) if _is_swing_low(lows, i, lookback)]


def detect_w_bottoms(
    df: pd.DataFrame,
    *,
    swing_lookback: int = 3,
    low_tolerance_pct: float = 0.003,
    min_bars_between_lows: int = 5,
    max_bars_between_lows: int = 60,
    require_neckline_break: bool = True,
) -> list[WBottomPattern]:
    """
    在 OHLCV DataFrame 上偵測 W 底型態。

    條件：
    1. 兩個相近的波段低點（價差在 low_tolerance_pct 內）
    2. 兩低點之間有明確的頸線高點
    3. 第二低點確認後，收盤突破頸線（可選）

    Parameters
    ----------
    df : DataFrame
        需含 open, high, low, close 欄位，index 為時間。
    swing_lookback : int
        左右各幾根 K 確認轉折。
    low_tolerance_pct : float
        兩低點允許的最大價差比例（0.003 = 0.3%）。
    min_bars_between_lows : int
        兩低點最少間隔 K 數。
    max_bars_between_lows : int
        兩低點最多間隔 K 數。
    require_neckline_break : bool
        是否要求收盤突破頸線才視為有效。
    """
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame 缺少欄位: {missing}")

    lows = df["low"].tolist()
    highs = df["high"].tolist()
    closes = df["close"].tolist()

    swing_lows = _find_swing_lows(lows, swing_lookback)
    patterns: list[WBottomPattern] = []

    for i, first_idx in enumerate(swing_lows):
        for second_idx in swing_lows[i + 1 :]:
            gap = second_idx - first_idx
            if gap < min_bars_between_lows:
                continue
            if gap > max_bars_between_lows:
                break

            first_low = lows[first_idx]
            second_low = lows[second_idx]
            avg_low = (first_low + second_low) / 2
            if avg_low == 0:
                continue
            if abs(first_low - second_low) / avg_low > low_tolerance_pct:
                continue

            # 頸線：兩低點之間的最高波段高點
            neckline_idx: int | None = None
            neckline_price = float("-inf")
            for j in range(first_idx + swing_lookback, second_idx - swing_lookback + 1):
                if _is_swing_high(highs, j, swing_lookback) and highs[j] > neckline_price:
                    neckline_idx = j
                    neckline_price = highs[j]

            if neckline_idx is None:
                continue

            breakout_idx: int | None = None
            if require_neckline_break:
                for k in range(second_idx + swing_lookback, len(closes)):
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
                    breakout_idx=breakout_idx,
                )
            )

    return _dedupe_patterns(patterns)


def _dedupe_patterns(patterns: Iterable[WBottomPattern]) -> list[WBottomPattern]:
    """同一突破 K 只保留風險報酬最佳的型態。"""
    by_breakout: dict[int, WBottomPattern] = {}
    for p in patterns:
        if p.breakout_idx is None:
            continue
        depth = p.neckline - min(p.first_low, p.second_low)
        risk = p.neckline - p.second_low
        rr = depth / risk if risk > 0 else 0
        existing = by_breakout.get(p.breakout_idx)
        if existing is None:
            by_breakout[p.breakout_idx] = p
            continue
        ex_depth = existing.neckline - min(existing.first_low, existing.second_low)
        ex_risk = existing.neckline - existing.second_low
        ex_rr = ex_depth / ex_risk if ex_risk > 0 else 0
        if rr > ex_rr:
            by_breakout[p.breakout_idx] = p
    return sorted(by_breakout.values(), key=lambda p: p.breakout_idx or 0)

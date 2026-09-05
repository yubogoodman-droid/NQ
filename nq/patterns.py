"""五分 K：跌破近三小時低點後，半小時內收盤站上 MA30。"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


LOOKBACK_BARS = 36  # 3 小時 × 12 根/小時
RECLAIM_BARS = 6  # 30 分鐘
MA30_PERIOD = 30


@dataclass(frozen=True)
class ReclaimPattern:
    """破三小時低點後，在時限內站上 MA30。"""

    break_idx: int
    entry_idx: int
    lookback_low: float
    break_low: float
    ma30: float

    @property
    def l1_idx(self) -> int:
        return self.break_idx

    @property
    def l2_idx(self) -> int:
        return self.break_idx

    @property
    def l3_idx(self) -> int:
        return self.entry_idx

    @property
    def neckline_idx(self) -> int:
        return self.break_idx

    @property
    def l1(self) -> float:
        return self.lookback_low

    @property
    def l2(self) -> float:
        return self.break_low

    @property
    def l3(self) -> float:
        return self.break_low

    @property
    def neckline(self) -> float:
        return self.lookback_low

    @property
    def reclaim_idx(self) -> int:
        return self.entry_idx

    @property
    def breakout_idx(self) -> int | None:
        return self.entry_idx

    @property
    def stop_loss(self) -> float:
        return self.break_low

    @property
    def target(self) -> float:
        """佔位；實際目標由策略用 2R 計算。"""
        return self.ma30


WBottomPattern = ReclaimPattern


def _sma(closes: list[float], idx: int, period: int) -> float | None:
    if idx + 1 < period:
        return None
    return sum(closes[idx - period + 1 : idx + 1]) / period


def detect_reclaims(
    df: pd.DataFrame,
    *,
    lookback_bars: int = LOOKBACK_BARS,
    reclaim_bars: int = RECLAIM_BARS,
    ma_period: int = MA30_PERIOD,
) -> list[ReclaimPattern]:
    """
    當根低點跌破「前 lookback_bars 根」的最低點，視為破三小時低。
    從破底那根起算 reclaim_bars 根內，收盤站上 MA30 則進場。
    """
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame 缺少欄位: {missing}")

    lows = df["low"].astype(float).tolist()
    closes = df["close"].astype(float).tolist()
    n = len(df)
    start = max(lookback_bars, ma_period)
    patterns: list[ReclaimPattern] = []
    i = start
    while i < n:
        lookback = min(lows[i - lookback_bars : i])
        fresh = lows[i] < lookback and lows[i - 1] >= lookback
        if not fresh:
            i += 1
            continue
        break_low = lows[i]
        found: ReclaimPattern | None = None
        last = min(i + reclaim_bars - 1, n - 1)
        for k in range(i, last + 1):
            ma30 = _sma(closes, k, ma_period)
            if ma30 is None:
                continue
            if closes[k] > ma30:
                found = ReclaimPattern(
                    break_idx=i,
                    entry_idx=k,
                    lookback_low=lookback,
                    break_low=min(lows[i : k + 1]),
                    ma30=ma30,
                )
                break
        if found:
            patterns.append(found)
            i = found.entry_idx + 1
        else:
            i = last + 1
    return patterns


def detect_w_bottoms(df: pd.DataFrame, **_: object) -> list[ReclaimPattern]:
    """相容舊名稱：改為三小時破底翻 MA30。"""
    return detect_reclaims(df)

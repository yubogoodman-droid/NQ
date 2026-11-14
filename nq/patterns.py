"""五分 K：跌破近三小時低點後，一小時內 MA5/MA20 多排站上 MA60。"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


LOOKBACK_BARS = 36  # 3 小時
RECLAIM_BARS = 12  # 1 小時
MA5_PERIOD = 5
MA20_PERIOD = 20
MA60_PERIOD = 60


@dataclass(frozen=True)
class ReclaimPattern:
    """破三小時低點後，時限內 MA5>MA20>MA60 且收盤站上 MA60。"""

    break_idx: int
    entry_idx: int
    lookback_low: float
    break_low: float
    ma5: float
    ma20: float
    ma60: float

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
        return self.ma60


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
    ma5_period: int = MA5_PERIOD,
    ma20_period: int = MA20_PERIOD,
    ma60_period: int = MA60_PERIOD,
) -> list[ReclaimPattern]:
    """
    當根低點跌破前 lookback_bars 根最低點。
    從破底起 reclaim_bars 根內，收盤站上 MA60 且 MA5 > MA20 > MA60 則進場。
    """
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame 缺少欄位: {missing}")

    lows = df["low"].astype(float).tolist()
    closes = df["close"].astype(float).tolist()
    n = len(df)
    start = max(lookback_bars, ma60_period)
    patterns: list[ReclaimPattern] = []
    i = start
    while i < n:
        lookback = min(lows[i - lookback_bars : i])
        fresh = lows[i] < lookback and lows[i - 1] >= lookback
        if not fresh:
            i += 1
            continue
        found: ReclaimPattern | None = None
        last = min(i + reclaim_bars - 1, n - 1)
        for k in range(i, last + 1):
            ma5 = _sma(closes, k, ma5_period)
            ma20 = _sma(closes, k, ma20_period)
            ma60 = _sma(closes, k, ma60_period)
            if ma5 is None or ma20 is None or ma60 is None:
                continue
            if closes[k] > ma60 and ma5 > ma20 > ma60:
                found = ReclaimPattern(
                    break_idx=i,
                    entry_idx=k,
                    lookback_low=lookback,
                    break_low=min(lows[i : k + 1]),
                    ma5=ma5,
                    ma20=ma20,
                    ma60=ma60,
                )
                break
        if found:
            patterns.append(found)
            i = found.entry_idx + 1
        else:
            i = last + 1
    return patterns


def detect_w_bottoms(df: pd.DataFrame, **_: object) -> list[ReclaimPattern]:
    return detect_reclaims(df)

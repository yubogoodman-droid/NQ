"""NQ 五分 K：破三小時低點後一小時內 MA5/MA20 多排站上 MA60 做多。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd

from nq.patterns import ReclaimPattern, detect_reclaims


class Side(str, Enum):
    LONG = "long"


@dataclass(frozen=True)
class Signal:
    timestamp: pd.Timestamp
    side: Side
    entry: float
    stop_loss: float
    target: float
    pattern: ReclaimPattern
    bar_idx: int

    @property
    def risk(self) -> float:
        return self.entry - self.stop_loss

    @property
    def reward(self) -> float:
        return self.target - self.entry


@dataclass
class NQWBottomStrategy:
    """
    五分 K 跌破近 3 小時低點後，1 小時內（12 根）
    收盤站上 MA60，且 MA5 > MA20 > MA60，做多。
    停損：破底低點下方 20 點
    停利：2R
    """

    lookback_bars: int = 36
    reclaim_bars: int = 12
    ma5_period: int = 5
    ma20_period: int = 20
    ma60_period: int = 60
    stop_below_break_points: float = 20.0
    reward_risk: float = 2.0
    tick_size: float = 0.25
    point_value: float = 20.0

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        patterns = detect_reclaims(
            df,
            lookback_bars=self.lookback_bars,
            reclaim_bars=self.reclaim_bars,
            ma5_period=self.ma5_period,
            ma20_period=self.ma20_period,
            ma60_period=self.ma60_period,
        )
        signals: list[Signal] = []
        for pattern in patterns:
            idx = pattern.entry_idx
            entry = self._round_tick(float(df["close"].iloc[idx]))
            stop = self._round_tick(pattern.break_low - self.stop_below_break_points)
            if entry <= stop:
                continue
            risk = entry - stop
            target = self._round_tick(entry + self.reward_risk * risk)
            signals.append(
                Signal(
                    timestamp=df.index[idx],
                    side=Side.LONG,
                    entry=entry,
                    stop_loss=stop,
                    target=target,
                    pattern=pattern,
                    bar_idx=idx,
                )
            )
        return signals

    def _round_tick(self, price: float) -> float:
        return round(price / self.tick_size) * self.tick_size

"""NQ 五分 K W 底進場策略。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd

from nq.patterns import WBottomPattern, detect_w_bottoms


class Side(str, Enum):
    LONG = "long"


@dataclass(frozen=True)
class Signal:
    """進場訊號。"""

    timestamp: pd.Timestamp
    side: Side
    entry: float
    stop_loss: float
    target: float
    pattern: WBottomPattern
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
    NQ（那斯達克期貨）五分 K W 底做多策略。

    進場：第二低點確認且收盤突破頸線
    停損：第二低點
    停利：量度漲幅（頸線 - 最低點）投射至突破點上方
    """

    swing_lookback: int = 3
    low_tolerance_pct: float = 0.001
    min_bars_between_lows: int = 5
    max_bars_between_lows: int = 60
    tick_size: float = 0.25
    point_value: float = 20.0  # NQ 每點 $20

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        patterns = detect_w_bottoms(
            df,
            swing_lookback=self.swing_lookback,
            low_tolerance_pct=self.low_tolerance_pct,
            min_bars_between_lows=self.min_bars_between_lows,
            max_bars_between_lows=self.max_bars_between_lows,
            require_neckline_break=True,
        )

        signals: list[Signal] = []
        for pattern in patterns:
            if pattern.breakout_idx is None:
                continue
            idx = pattern.breakout_idx
            entry = self._round_tick(df["close"].iloc[idx])
            stop = self._round_tick(pattern.stop_loss)
            target = self._round_tick(pattern.target)

            if entry <= stop:
                continue

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

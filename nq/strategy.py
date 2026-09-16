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

    預設對齊券商圖那種：先有一段殺勢，打出兩個相近谷，
    同一根 Globex session 內很快收盤破頸線才進。淺雙底、隔夜才破的不算。
    停損：第二低點；停利：量度漲幅。
    """

    swing_lookback: int = 3
    low_tolerance_pct: float = 0.001
    min_bars_between_lows: int = 8
    max_bars_between_lows: int = 48
    tick_size: float = 0.25
    point_value: float = 20.0  # NQ 每點 $20
    # 對齊 2026-09-15 04:10/05:15 → 05:35 破頸線（L1 29231 / L2 29236.5）
    min_prior_drop_pct: float = 0.0055
    prior_lookback: int = 36
    min_dump_bars: int = 10
    min_neck_pct: float = 0.0016
    min_right_leg_pct: float = 0.0015
    max_neck_retrace: float = 0.40
    max_mid_swing_lows: int = 1
    max_bars_to_break: int = 12
    min_neck_offset: int = 2
    min_right_bars: int = 3
    require_same_globex_session: bool = True

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        patterns = detect_w_bottoms(
            df,
            swing_lookback=self.swing_lookback,
            low_tolerance_pct=self.low_tolerance_pct,
            min_bars_between_lows=self.min_bars_between_lows,
            max_bars_between_lows=self.max_bars_between_lows,
            require_neckline_break=True,
            min_prior_drop_pct=self.min_prior_drop_pct,
            prior_lookback=self.prior_lookback,
            min_dump_bars=self.min_dump_bars,
            min_neck_pct=self.min_neck_pct,
            min_right_leg_pct=self.min_right_leg_pct,
            max_neck_retrace=self.max_neck_retrace,
            max_mid_swing_lows=self.max_mid_swing_lows,
            max_bars_to_break=self.max_bars_to_break,
            min_neck_offset=self.min_neck_offset,
            min_right_bars=self.min_right_bars,
            require_same_globex_session=self.require_same_globex_session,
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

    @classmethod
    def loose(cls) -> "NQWBottomStrategy":
        """舊版：只看兩低點價差與破頸線，不濾殺勢／形狀。"""
        return cls(
            min_bars_between_lows=5,
            max_bars_between_lows=60,
            min_prior_drop_pct=0.0,
            min_dump_bars=0,
            min_neck_pct=0.0,
            min_right_leg_pct=0.0,
            max_neck_retrace=9.0,
            max_mid_swing_lows=99,
            max_bars_to_break=60,
            min_neck_offset=0,
            min_right_bars=0,
            require_same_globex_session=False,
        )

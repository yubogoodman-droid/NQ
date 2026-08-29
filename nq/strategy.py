"""NQ 五分 K 破底 W 底進場策略。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd

from nq.patterns import WBottomPattern, detect_w_bottoms


class Side(str, Enum):
    LONG = "long"


@dataclass(frozen=True)
class Signal:
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
    NQ 五分 K 破底 W 底做多。

    L1 左腳 → 反彈 → L2 中間破底 → 收復 → L3 右腳收盤進場
    停損：L3 下方 20 點
    停利：頸線 + (頸線 − L2)
    均線糾結（MA5/10/20/60/120/200 高低差過小）不進場
    MA200 下彎時，進場不能離 MA200 太遠
    """

    swing_lookback: int = 3
    low_tolerance_pct: float = 0.001
    min_bars_between_lows: int = 8
    max_bars_between_lows: int = 24
    max_pattern_hours: float = 2.0
    min_spring_pct: float = 0.001
    min_spring_points: float = 25.0
    min_bounce_pct: float = 0.001
    max_reclaim_bars: int = 36
    max_breakout_bars: int = 36
    stop_below_l3_points: float = 20.0
    min_ma_span_points: float = 40.0
    ma_periods: tuple[int, ...] = (5, 10, 20, 60, 120, 200)
    ma200_slope_bars: int = 12
    max_dist_falling_ma200: float = 60.0
    tick_size: float = 0.25
    point_value: float = 20.0

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        patterns = detect_w_bottoms(
            df,
            swing_lookback=self.swing_lookback,
            low_tolerance_pct=self.low_tolerance_pct,
            min_bars_between_lows=self.min_bars_between_lows,
            max_bars_between_lows=self.max_bars_between_lows,
            max_pattern_hours=self.max_pattern_hours,
            min_spring_pct=self.min_spring_pct,
            min_spring_points=self.min_spring_points,
            min_bounce_pct=self.min_bounce_pct,
            max_reclaim_bars=self.max_reclaim_bars,
            max_breakout_bars=self.max_breakout_bars,
            require_neckline_break=False,
        )
        signals: list[Signal] = []
        for pattern in patterns:
            idx = pattern.l3_idx
            entry = self._round_tick(float(df["close"].iloc[idx]))
            stop = self._round_tick(pattern.l3 - self.stop_below_l3_points)
            target = self._round_tick(pattern.target)
            if entry <= stop:
                continue
            if self._mas_tangled(df, idx):
                continue
            if self._too_far_from_falling_ma200(df, idx, entry):
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

    def _mas_tangled(self, df: pd.DataFrame, idx: int) -> bool:
        """六條均線高低差過小 = 糾結盤整，不進場。K 數不夠算 MA60 時不濾。"""
        close = df["close"].astype(float)
        vals: list[float] = []
        for period in self.ma_periods:
            if idx + 1 < period:
                continue
            vals.append(float(close.iloc[idx - period + 1 : idx + 1].mean()))
        if len(vals) < 4:
            return False
        return max(vals) - min(vals) < self.min_ma_span_points

    def _too_far_from_falling_ma200(self, df: pd.DataFrame, idx: int, entry: float) -> bool:
        """MA200 下彎時，進場價離 MA200 過遠則跳過。"""
        need = 200 + self.ma200_slope_bars
        if idx + 1 < need:
            return False
        close = df["close"].astype(float)
        ma_now = float(close.iloc[idx - 199 : idx + 1].mean())
        prev = idx - self.ma200_slope_bars
        ma_prev = float(close.iloc[prev - 199 : prev + 1].mean())
        if ma_now >= ma_prev:
            return False
        return abs(entry - ma_now) > self.max_dist_falling_ma200

    def _round_tick(self, price: float) -> float:
        return round(price / self.tick_size) * self.tick_size

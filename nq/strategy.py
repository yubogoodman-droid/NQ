"""NQ 五分 K W 底進場策略。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd

from nq.patterns import WBottomPattern, detect_w_bottoms
from nq.spring import FakeBreakdownPattern, detect_fake_breakdowns


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
    pattern: WBottomPattern | FakeBreakdownPattern
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


@dataclass
class FakeBreakdownStrategy:
    """
    假跌破後上拉（Spring）做多策略。

    進場：盤整箱體被假跌破、迅速站回後，收盤突破頸線（盤整收盤高）且放量
    停損：假跌破最低點
    停利：進場價 + reward_r × 風險（預設 2R）
    """

    range_bars: int = 18
    range_max_pct: float = 0.02
    ma_cluster_pct: float = 0.012
    min_break_pct: float = 0.005
    max_break_pct: float = 0.03
    max_spring_bars: int = 15
    max_reclaim_bars: int = 10
    max_breakout_bars: int = 24
    neckline_frac: float = 0.8
    breakout_vol_mult: float = 1.2
    spring_vol_max_mult: float = 1.35
    require_volume: bool = True
    reward_r: float = 2.0
    tick_size: float = 0.5
    point_value: float = 1.0
    same_session: bool = True
    skip_open_minutes: int = 5

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        patterns = detect_fake_breakdowns(
            df,
            range_bars=self.range_bars,
            range_max_pct=self.range_max_pct,
            ma_cluster_pct=self.ma_cluster_pct,
            min_break_pct=self.min_break_pct,
            max_break_pct=self.max_break_pct,
            max_spring_bars=self.max_spring_bars,
            max_reclaim_bars=self.max_reclaim_bars,
            max_breakout_bars=self.max_breakout_bars,
            neckline_frac=self.neckline_frac,
            breakout_vol_mult=self.breakout_vol_mult,
            spring_vol_max_mult=self.spring_vol_max_mult,
            require_volume=self.require_volume,
            same_session=self.same_session,
            skip_open_minutes=self.skip_open_minutes,
        )

        signals: list[Signal] = []
        for pattern in patterns:
            if pattern.breakout_idx is None:
                continue
            idx = pattern.breakout_idx
            entry = self._round_tick(float(df["close"].iloc[idx]))
            stop = self._round_tick(pattern.stop_loss)
            risk = entry - stop
            if risk <= 0:
                continue
            target = self._round_tick(entry + self.reward_r * risk)

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

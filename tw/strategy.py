"""5/10/20 空頭排列 + 收盤跌破 MA200 做空進場。"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TwSignal:
    timestamp: pd.Timestamp
    ticker: str
    name: str
    entry: float
    ma5: float
    ma10: float
    ma20: float
    ma200: float
    bar_idx: int


@dataclass
class TwMaShortStrategy:
    """
    進場（做空）：
    1. MA5 < MA10 < MA20（空頭排列）
    2. 當日收盤跌破 MA200（前一日收盤 >= MA200，當日收盤 < MA200）
    3. 當日收盤價 <= max_price
    """

    max_price: float = 600.0

    def generate_signals(
        self,
        df: pd.DataFrame,
        *,
        ticker: str,
        name: str = "",
        eligible: pd.Series | None = None,
    ) -> list[TwSignal]:
        if df.empty or len(df) < 201:
            return []

        close = df["close"]
        ma5 = close.rolling(5, min_periods=5).mean()
        ma10 = close.rolling(10, min_periods=10).mean()
        ma20 = close.rolling(20, min_periods=20).mean()
        ma200 = close.rolling(200, min_periods=200).mean()

        bearish = (ma5 < ma10) & (ma10 < ma20)
        breakdown = (close < ma200) & (close.shift(1) >= ma200.shift(1))
        cheap = close <= self.max_price
        signal_mask = bearish & breakdown & cheap
        if eligible is not None:
            elig = eligible.reindex(df.index).fillna(False).astype(bool)
            signal_mask = signal_mask & elig

        signals: list[TwSignal] = []
        hits = signal_mask.fillna(False)
        for ts, flag in hits.items():
            if not flag:
                continue
            idx = df.index.get_loc(ts)
            if isinstance(idx, slice):
                idx = idx.start
            signals.append(
                TwSignal(
                    timestamp=pd.Timestamp(ts),
                    ticker=ticker,
                    name=name,
                    entry=float(close.iloc[idx]),
                    ma5=float(ma5.iloc[idx]),
                    ma10=float(ma10.iloc[idx]),
                    ma20=float(ma20.iloc[idx]),
                    ma200=float(ma200.iloc[idx]),
                    bar_idx=int(idx),
                )
            )
        return signals

"""一分 K：MA5/10/20 多頭排列，且收盤剛站上 MA200。"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


MA_FAST = 5
MA_MID = 10
MA_SLOW = 20
MA_LONG = 200


@dataclass(frozen=True)
class AlertSnapshot:
    timestamp: pd.Timestamp
    close: float
    prev_close: float
    ma5: float
    ma10: float
    ma20: float
    ma200: float
    prev_ma200: float

    @property
    def bullish_aligned(self) -> bool:
        return self.ma5 > self.ma10 > self.ma20

    @property
    def crossed_above_ma200(self) -> bool:
        return self.close > self.ma200 and self.prev_close <= self.prev_ma200


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["close"]
    out["ma5"] = close.rolling(MA_FAST, min_periods=MA_FAST).mean()
    out["ma10"] = close.rolling(MA_MID, min_periods=MA_MID).mean()
    out["ma20"] = close.rolling(MA_SLOW, min_periods=MA_SLOW).mean()
    out["ma200"] = close.rolling(MA_LONG, min_periods=MA_LONG).mean()
    return out


def is_ma200_breakout_bullish(df: pd.DataFrame) -> AlertSnapshot | None:
    """
    最新一根：MA5 > MA10 > MA20，收盤站上 MA200；
    前一根收盤尚未站上（相對該根的 MA200）。
    """
    if df is None or len(df) < MA_LONG + 1:
        return None
    work = add_moving_averages(df)
    last = work.iloc[-1]
    prev = work.iloc[-2]
    needed = ("ma5", "ma10", "ma20", "ma200")
    if any(pd.isna(last[col]) or pd.isna(prev[col]) for col in needed):
        return None

    snapshot = AlertSnapshot(
        timestamp=work.index[-1],
        close=float(last["close"]),
        prev_close=float(prev["close"]),
        ma5=float(last["ma5"]),
        ma10=float(last["ma10"]),
        ma20=float(last["ma20"]),
        ma200=float(last["ma200"]),
        prev_ma200=float(prev["ma200"]),
    )
    if snapshot.bullish_aligned and snapshot.crossed_above_ma200:
        return snapshot
    return None

"""五分 K：MA5 < MA10 < MA20 空頭排列，當根收盤跌破 MA200，且小時K在 MA20 之下。"""

from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd

from tw.kline import resample_ohlcv


MA_FAST = 5
MA_MID = 10
MA_SLOW = 20
MA_LONG = 200
H1_MA = 20


@dataclass(frozen=True)
class AlertSnapshot:
    timestamp: pd.Timestamp
    close: float
    prev_close: float
    ma5: float
    ma10: float
    ma20: float
    ma200: float
    prev_ma5: float
    prev_ma10: float
    prev_ma20: float
    prev_ma200: float
    h1_close: float | None = None
    h1_ma20: float | None = None

    @property
    def bearish_aligned(self) -> bool:
        return self.ma5 < self.ma10 < self.ma20

    @property
    def crossed_below_ma200(self) -> bool:
        return self.close < self.ma200 and self.prev_close >= self.prev_ma200

    @property
    def close_below_short_mas(self) -> bool:
        return self.close < self.ma5 and self.close < self.ma10 and self.close < self.ma20

    @property
    def hourly_close_below_ma20(self) -> bool:
        return (
            self.h1_close is not None
            and self.h1_ma20 is not None
            and self.h1_close < self.h1_ma20
        )


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["close"]
    out["ma5"] = close.rolling(MA_FAST, min_periods=MA_FAST).mean()
    out["ma10"] = close.rolling(MA_MID, min_periods=MA_MID).mean()
    out["ma20"] = close.rolling(MA_SLOW, min_periods=MA_SLOW).mean()
    out["ma200"] = close.rolling(MA_LONG, min_periods=MA_LONG).mean()
    return out


def hourly_close_and_ma20(five_min: pd.DataFrame) -> tuple[float, float] | None:
    """用截至目前的五分K合成小時K，回傳（小時收盤, 小時MA20）。"""
    hourly = resample_ohlcv(five_min, "1h")
    if len(hourly) < H1_MA or "close" not in hourly.columns:
        return None
    ma20 = hourly["close"].rolling(H1_MA, min_periods=H1_MA).mean()
    if pd.isna(ma20.iloc[-1]) or pd.isna(hourly["close"].iloc[-1]):
        return None
    return float(hourly["close"].iloc[-1]), float(ma20.iloc[-1])


def iter_5m_ma200_short_alerts(
    df: pd.DataFrame,
    *,
    since: pd.Timestamp | None = None,
    until: pd.Timestamp | None = None,
    latest_only: bool = False,
) -> list[AlertSnapshot]:
    """同一交易日連續五分 K：MA5 < MA10 < MA20，當根收盤剛跌破 MA200，且小時K在 MA20 之下。"""
    if df is None or len(df) < MA_LONG + 1:
        return []
    work = add_moving_averages(df)
    hits: list[AlertSnapshot] = []
    start = MA_LONG
    if since is not None:
        matched = False
        for i, ts in enumerate(work.index):
            if ts >= since:
                start = max(start, i)
                matched = True
                break
        if not matched:
            return []
    for i in range(start, len(work)):
        ts = work.index[i]
        if until is not None and ts > until:
            break
        snap = _snapshot_at(work, i)
        if snap is None:
            continue
        prev_ts = work.index[i - 1]
        if pd.Timestamp(ts).date() != pd.Timestamp(prev_ts).date():
            continue
        if not (snap.bearish_aligned and snap.crossed_below_ma200):
            continue
        hourly = hourly_close_and_ma20(work.iloc[: i + 1])
        if hourly is None:
            continue
        h1_close, h1_ma20 = hourly
        if h1_close >= h1_ma20:
            continue
        hits.append(replace(snap, h1_close=h1_close, h1_ma20=h1_ma20))
    if latest_only and hits:
        return hits[-1:]
    return hits


def _snapshot_at(work: pd.DataFrame, idx: int) -> AlertSnapshot | None:
    if idx < 1:
        return None
    last = work.iloc[idx]
    prev = work.iloc[idx - 1]
    needed = ("ma5", "ma10", "ma20", "ma200")
    if any(pd.isna(last[col]) or pd.isna(prev[col]) for col in needed):
        return None
    return AlertSnapshot(
        timestamp=work.index[idx],
        close=float(last["close"]),
        prev_close=float(prev["close"]),
        ma5=float(last["ma5"]),
        ma10=float(last["ma10"]),
        ma20=float(last["ma20"]),
        ma200=float(last["ma200"]),
        prev_ma5=float(prev["ma5"]),
        prev_ma10=float(prev["ma10"]),
        prev_ma20=float(prev["ma20"]),
        prev_ma200=float(prev["ma200"]),
    )

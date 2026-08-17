"""一分 K：MA5/10/20 多頭排列，且收盤剛站上 MA200。"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


MA_FAST = 5
MA_MID = 10
MA_SLOW = 20
MA_LONG = 200
# 09:00–09:04 開盤跳空不算「當下那根站上 200」。
ENTRY_AFTER_HOUR = 9
ENTRY_AFTER_MINUTE = 5
MAX_BAR_GAP = pd.Timedelta(minutes=2)


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

    @property
    def ma_span_pct(self) -> float:
        """MA5 與 MA20 的距離／收盤。太小代表均線糾結。"""
        if self.close <= 0:
            return 0.0
        return (self.ma5 - self.ma20) / self.close

    @property
    def ma20_ma200_gap_pct(self) -> float:
        """MA20 與 MA200 的距離／收盤。太大代表兩條線差太遠（像華新科）。"""
        if self.close <= 0:
            return 0.0
        return abs(self.ma200 - self.ma20) / self.close


def is_intraday_entry_bar(prev_ts: pd.Timestamp, ts: pd.Timestamp) -> bool:
    """同一根進場：前一根必須是同一交易日、連續的一分K，且已過開盤跳空時段。"""
    prev = pd.Timestamp(prev_ts)
    cur = pd.Timestamp(ts)
    if prev.tzinfo is not None and cur.tzinfo is None:
        cur = cur.tz_localize(prev.tzinfo)
    elif cur.tzinfo is not None and prev.tzinfo is None:
        prev = prev.tz_localize(cur.tzinfo)
    if cur.date() != prev.date():
        return False
    if cur.hour < ENTRY_AFTER_HOUR or (
        cur.hour == ENTRY_AFTER_HOUR and cur.minute < ENTRY_AFTER_MINUTE
    ):
        return False
    if cur - prev > MAX_BAR_GAP:
        return False
    return True


def mas_are_open(snapshot: AlertSnapshot, min_span: float = 0.002) -> bool:
    """多頭排列且 MA5–MA20 拉開到 min_span 以上（預設 0.2%）。"""
    return snapshot.bullish_aligned and snapshot.ma_span_pct >= min_span


def ma20_near_ma200(snapshot: AlertSnapshot, max_gap: float = 0.015) -> bool:
    """MA20 與 MA200 不能差太遠（預設 1.5%）。"""
    return snapshot.ma20_ma200_gap_pct <= max_gap


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["close"]
    out["ma5"] = close.rolling(MA_FAST, min_periods=MA_FAST).mean()
    out["ma10"] = close.rolling(MA_MID, min_periods=MA_MID).mean()
    out["ma20"] = close.rolling(MA_SLOW, min_periods=MA_SLOW).mean()
    out["ma200"] = close.rolling(MA_LONG, min_periods=MA_LONG).mean()
    return out


def is_ma200_breakout_bullish(df: pd.DataFrame) -> AlertSnapshot | None:
    """最新一根：MA5 > MA10 > MA20，收盤站上 MA200，且前一根尚未站上。"""
    return latest_ma200_breakout_bullish(df, latest_only=True)


def latest_ma200_breakout_bullish(
    df: pd.DataFrame,
    *,
    since: pd.Timestamp | None = None,
    until: pd.Timestamp | None = None,
    latest_only: bool = False,
) -> AlertSnapshot | None:
    """
    回傳符合條件的進場那一根：MA5>MA10>MA20，且這根收盤剛站上 MA200。
    latest_only=True 只看最後一根（盤中 watch）；否則找 since～until 裡第一根。
    隔夜跳空、開盤前 5 分鐘不算。
    """
    if df is None or len(df) < MA_LONG + 1:
        return None
    work = add_moving_averages(df)
    if latest_only:
        prev_ts = work.index[-2]
        ts = work.index[-1]
        if not is_intraday_entry_bar(prev_ts, ts):
            return None
        snap = _snapshot_at(work, len(work) - 1)
        if snap is None:
            return None
        if since is not None and snap.timestamp < since:
            return None
        if until is not None and snap.timestamp > until:
            return None
        if snap.bullish_aligned and snap.crossed_above_ma200:
            return snap
        return None

    start = MA_LONG
    if since is not None:
        matched = False
        for i, ts in enumerate(work.index):
            if ts >= since:
                start = max(MA_LONG, i)
                matched = True
                break
        if not matched:
            return None

    for i in range(start, len(work)):
        prev_ts = work.index[i - 1]
        ts = work.index[i]
        if until is not None and ts > until:
            break
        if not is_intraday_entry_bar(prev_ts, ts):
            continue
        snap = _snapshot_at(work, i)
        if snap and snap.bullish_aligned and snap.crossed_above_ma200:
            return snap
    return None


def ma200_at(
    df: pd.DataFrame,
    ts: pd.Timestamp | None = None,
    *,
    floor: str | None = None,
) -> tuple[float, float] | None:
    """回傳指定時間（或最新一根）的收盤與 MA200；資料不足則 None。"""
    if df is None or df.empty:
        return None
    work = add_moving_averages(df)
    if ts is None:
        loc = len(work) - 1
    else:
        mark = pd.Timestamp(ts)
        if floor:
            mark = mark.floor(floor)
        loc = int(work.index.get_indexer([mark], method="nearest")[0])
        if loc < 0:
            return None
    row = work.iloc[loc]
    if pd.isna(row.get("ma200")):
        return None
    return float(row["close"]), float(row["ma200"])


def ma200_gap_pct(
    df: pd.DataFrame,
    ts: pd.Timestamp | None = None,
    *,
    floor: str | None = None,
) -> float | None:
    """(收盤 − MA200) / MA200。資料不足或 MA200 ≤ 0 則 None。"""
    pair = ma200_at(df, ts, floor=floor)
    if pair is None or pair[1] <= 0:
        return None
    return (pair[0] - pair[1]) / pair[1]


def close_above_ma200(
    df: pd.DataFrame,
    ts: pd.Timestamp | None = None,
    *,
    floor: str | None = None,
    min_gap: float = 0.0,
) -> bool:
    gap = ma200_gap_pct(df, ts, floor=floor)
    return gap is not None and gap >= min_gap and gap > 0


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
        prev_ma200=float(prev["ma200"]),
    )

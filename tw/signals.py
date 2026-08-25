"""一分 K：MA5>MA10>MA20 多頭排列，收盤從下穿上 MA240 就通知。"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


MA_FAST = 5
MA_MID = 10
MA_SLOW = 20
MA_LONG = 240
MAX_BAR_GAP = pd.Timedelta(minutes=2)


@dataclass(frozen=True)
class AlertSnapshot:
    timestamp: pd.Timestamp
    close: float
    prev_close: float
    ma5: float
    ma10: float
    ma20: float
    ma240: float
    prev_ma240: float

    @property
    def bullish_aligned(self) -> bool:
        return self.ma5 > self.ma10 > self.ma20

    @property
    def crossed_above_ma240(self) -> bool:
        return self.close > self.ma240 and self.prev_close <= self.prev_ma240

    @property
    def ma_span_pct(self) -> float:
        if self.close <= 0:
            return 0.0
        return (self.ma5 - self.ma20) / self.close

    @property
    def ma20_ma240_gap_pct(self) -> float:
        if self.close <= 0:
            return 0.0
        return abs(self.ma240 - self.ma20) / self.close


def is_intraday_entry_bar(prev_ts: pd.Timestamp, ts: pd.Timestamp) -> bool:
    """前一根必須是同一交易日、連續的一分 K（隔夜跳空不算）。"""
    prev = pd.Timestamp(prev_ts)
    cur = pd.Timestamp(ts)
    if prev.tzinfo is not None and cur.tzinfo is None:
        cur = cur.tz_localize(prev.tzinfo)
    elif cur.tzinfo is not None and prev.tzinfo is None:
        prev = prev.tz_localize(cur.tzinfo)
    if cur.date() != prev.date():
        return False
    if cur - prev > MAX_BAR_GAP:
        return False
    return True


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["close"]
    out["ma5"] = close.rolling(MA_FAST, min_periods=MA_FAST).mean()
    out["ma10"] = close.rolling(MA_MID, min_periods=MA_MID).mean()
    out["ma20"] = close.rolling(MA_SLOW, min_periods=MA_SLOW).mean()
    out["ma240"] = close.rolling(MA_LONG, min_periods=MA_LONG).mean()
    return out


def is_ma240_breakout_bullish(df: pd.DataFrame) -> AlertSnapshot | None:
    """最新一根就是多頭排列＋剛站上 MA240。"""
    return latest_ma240_breakout_bullish(df, latest_only=True)


def latest_ma240_breakout_bullish(
    df: pd.DataFrame,
    *,
    since: pd.Timestamp | None = None,
    until: pd.Timestamp | None = None,
    latest_only: bool = False,
    min_ma_span: float = 0.0,
    min_ma20_ma240_gap: float = 0.0,
    max_ma20_ma240_gap: float | None = None,
) -> AlertSnapshot | None:
    """
    通知那一根：MA5>MA10>MA20，且收盤從 MA240 下面（含等於）穿上。
    latest_only 只看最後一根（watch）。隔夜跳空不算。
    多餘參數保留是為了舊呼叫端相容，不再使用。
    """
    del min_ma_span, min_ma20_ma240_gap, max_ma20_ma240_gap
    if df is None or len(df) < MA_LONG + 1:
        return None
    work = add_moving_averages(df)
    if latest_only:
        loc = len(work) - 1
        snap = _cross_at(work, loc)
        if snap is None:
            return None
        if since is not None and snap.timestamp < since:
            return None
        if until is not None and snap.timestamp > until:
            return None
        return snap

    start = MA_LONG
    if since is not None:
        matched = False
        for i, ts in enumerate(work.index):
            if ts >= since:
                start = max(start, i)
                matched = True
                break
        if not matched:
            return None

    for i in range(start, len(work)):
        ts = work.index[i]
        if until is not None and ts > until:
            break
        snap = _cross_at(work, i)
        if snap is not None:
            return snap
    return None


def _cross_at(work: pd.DataFrame, idx: int) -> AlertSnapshot | None:
    """idx 為剛站上 MA240 的那一根。"""
    if idx < 1:
        return None
    if not is_intraday_entry_bar(work.index[idx - 1], work.index[idx]):
        return None
    snap = _snapshot_at(work, idx)
    if snap is None:
        return None
    if not snap.crossed_above_ma240:
        return None
    if not snap.bullish_aligned:
        return None
    return snap


def ma240_at(
    df: pd.DataFrame,
    ts: pd.Timestamp | None = None,
    *,
    floor: str | None = None,
) -> tuple[float, float] | None:
    """回傳指定時間（或最新一根）的收盤與 MA240；資料不足則 None。"""
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
    if pd.isna(row.get("ma240")):
        return None
    return float(row["close"]), float(row["ma240"])


def ma240_gap_pct(
    df: pd.DataFrame,
    ts: pd.Timestamp | None = None,
    *,
    floor: str | None = None,
) -> float | None:
    """(收盤 − MA240) / MA240。資料不足或 MA240 ≤ 0 則 None。"""
    pair = ma240_at(df, ts, floor=floor)
    if pair is None or pair[1] <= 0:
        return None
    return (pair[0] - pair[1]) / pair[1]


def close_above_ma240(
    df: pd.DataFrame,
    ts: pd.Timestamp | None = None,
    *,
    floor: str | None = None,
    min_gap: float = 0.0,
) -> bool:
    gap = ma240_gap_pct(df, ts, floor=floor)
    return gap is not None and gap >= min_gap and gap > 0


def _snapshot_at(work: pd.DataFrame, idx: int) -> AlertSnapshot | None:
    if idx < 1:
        return None
    last = work.iloc[idx]
    prev = work.iloc[idx - 1]
    needed = ("ma5", "ma10", "ma20", "ma240")
    if any(pd.isna(last[col]) or pd.isna(prev[col]) for col in needed):
        return None
    return AlertSnapshot(
        timestamp=work.index[idx],
        close=float(last["close"]),
        prev_close=float(prev["close"]),
        ma5=float(last["ma5"]),
        ma10=float(last["ma10"]),
        ma20=float(last["ma20"]),
        ma240=float(last["ma240"]),
        prev_ma240=float(prev["ma240"]),
    )

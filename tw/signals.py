"""五分／十五分 K：多方剛站上 MA200，或空方剛跌破 MA200。"""

from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd

from tw.kline import resample_ohlcv


MA_FAST = 5
MA_MID = 10
MA_SLOW = 20
MA_MED = 60
MA_LONG = 200
H1_MA = 20
# MA5 相對 MA20 至少拉開這麼多（％），否則算糾結。
MIN_RIBBON_FAN_PCT = 0.50
# MA5–MA10、MA10–MA20 各自相對收盤至少這麼多（％），避免其中兩條黏在一起。
MIN_RIBBON_GAP_PCT = 0.10
# 十五分收盤要在 MA200 上，而且連續至少這麼久。
M15_ABOVE_MA200_MINUTES = 30
M15_BAR_MINUTES = 15


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
    h1_ma5: float | None = None
    h1_ma10: float | None = None
    h1_ma20: float | None = None
    m15_close: float | None = None
    m15_ma5: float | None = None
    m15_ma10: float | None = None
    m15_ma20: float | None = None
    m15_ma200: float | None = None
    m15_above_ma200_minutes: int | None = None
    side: str = "long"

    @property
    def bullish_aligned(self) -> bool:
        return self.ma5 > self.ma10 > self.ma20

    @property
    def bearish_aligned(self) -> bool:
        return self.ma5 < self.ma10 < self.ma20

    @property
    def mas_rising(self) -> bool:
        return self.ma5 > self.prev_ma5 and self.ma10 > self.prev_ma10 and self.ma20 > self.prev_ma20

    @property
    def mas_falling(self) -> bool:
        return self.ma5 < self.prev_ma5 and self.ma10 < self.prev_ma10 and self.ma20 < self.prev_ma20

    @property
    def ribbon_fan_pct(self) -> float:
        if self.ma20 == 0:
            return 0.0
        return (self.ma5 / self.ma20 - 1.0) * 100.0

    @property
    def gap_5_10_pct(self) -> float:
        if not self.close:
            return 0.0
        return (self.ma5 - self.ma10) / self.close * 100.0

    @property
    def gap_10_20_pct(self) -> float:
        if not self.close:
            return 0.0
        return (self.ma10 - self.ma20) / self.close * 100.0

    @property
    def ribbon_fanned(self) -> bool:
        """多頭排列：MA5 > MA10 > MA20，且三條都比前一根高。"""
        return self.bullish_aligned and self.mas_rising

    @property
    def ribbon_down(self) -> bool:
        """空頭排列：MA5 < MA10 < MA20，且三條都比前一根低。"""
        return self.bearish_aligned and self.mas_falling

    @property
    def crossed_above_ma200(self) -> bool:
        return self.close > self.ma200 and self.prev_close <= self.prev_ma200

    @property
    def crossed_below_ma200(self) -> bool:
        return self.close < self.ma200 and self.prev_close >= self.prev_ma200

    @property
    def close_above_all_mas(self) -> bool:
        return (
            self.close > self.ma5
            and self.close > self.ma10
            and self.close > self.ma20
            and self.close > self.ma200
        )

    @property
    def close_below_all_mas(self) -> bool:
        return (
            self.close < self.ma5
            and self.close < self.ma10
            and self.close < self.ma20
            and self.close < self.ma200
        )

    @property
    def hourly_close_above_ma20(self) -> bool:
        return (
            self.h1_close is not None
            and self.h1_ma20 is not None
            and self.h1_close > self.h1_ma20
        )

    @property
    def hourly_close_above_short_mas(self) -> bool:
        return (
            self.h1_close is not None
            and self.h1_ma5 is not None
            and self.h1_ma10 is not None
            and self.h1_ma20 is not None
            and self.h1_close > self.h1_ma5
            and self.h1_close > self.h1_ma10
            and self.h1_close > self.h1_ma20
        )

    @property
    def close_above_15m_mas(self) -> bool:
        return (
            self.m15_close is not None
            and self.m15_ma5 is not None
            and self.m15_ma10 is not None
            and self.m15_ma20 is not None
            and self.m15_close > self.m15_ma5
            and self.m15_close > self.m15_ma10
            and self.m15_close > self.m15_ma20
        )

    @property
    def fifteen_above_ma200_half_hour(self) -> bool:
        return (
            self.m15_close is not None
            and self.m15_ma200 is not None
            and self.m15_above_ma200_minutes is not None
            and self.m15_close > self.m15_ma200
            and self.m15_above_ma200_minutes >= M15_ABOVE_MA200_MINUTES
        )


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["close"]
    out["ma5"] = close.rolling(MA_FAST, min_periods=MA_FAST).mean()
    out["ma10"] = close.rolling(MA_MID, min_periods=MA_MID).mean()
    out["ma20"] = close.rolling(MA_SLOW, min_periods=MA_SLOW).mean()
    out["ma60"] = close.rolling(MA_MED, min_periods=MA_MED).mean()
    out["ma200"] = close.rolling(MA_LONG, min_periods=MA_LONG).mean()
    return out


def hourly_close_and_mas(five_min: pd.DataFrame) -> tuple[float, float, float, float] | None:
    """用截至目前的五分K合成小時K，回傳（收盤, MA5, MA10, MA20）。"""
    hourly = resample_ohlcv(five_min, "1h")
    if len(hourly) < H1_MA or "close" not in hourly.columns:
        return None
    close = hourly["close"]
    ma5 = close.rolling(MA_FAST, min_periods=MA_FAST).mean()
    ma10 = close.rolling(MA_MID, min_periods=MA_MID).mean()
    ma20 = close.rolling(H1_MA, min_periods=H1_MA).mean()
    last = (close.iloc[-1], ma5.iloc[-1], ma10.iloc[-1], ma20.iloc[-1])
    if any(pd.isna(v) for v in last):
        return None
    return float(last[0]), float(last[1]), float(last[2]), float(last[3])


def hourly_close_and_ma20(five_min: pd.DataFrame) -> tuple[float, float] | None:
    """用截至目前的五分K合成小時K，回傳（小時收盤, 小時MA20）。"""
    hourly = hourly_close_and_mas(five_min)
    if hourly is None:
        return None
    return hourly[0], hourly[3]


@dataclass(frozen=True)
class FifteenSnapshot:
    close: float
    ma5: float
    ma10: float
    ma20: float
    ma200: float
    above_ma200_minutes: int


def fifteen_close_and_mas(five_min: pd.DataFrame) -> FifteenSnapshot | None:
    """用截至目前的五分K合成十五分K，含短均與 MA200，以及收盤在 MA200 上多久。"""
    m15 = resample_ohlcv(five_min, "15min")
    if len(m15) < MA_LONG or "close" not in m15.columns:
        return None
    work = add_moving_averages(m15)
    last = work.iloc[-1]
    needed = ("close", "ma5", "ma10", "ma20", "ma200")
    if any(pd.isna(last[col]) for col in needed):
        return None
    minutes = _minutes_above_ma200(work, pd.Timestamp(five_min.index[-1]))
    return FifteenSnapshot(
        close=float(last["close"]),
        ma5=float(last["ma5"]),
        ma10=float(last["ma10"]),
        ma20=float(last["ma20"]),
        ma200=float(last["ma200"]),
        above_ma200_minutes=minutes,
    )


def _minutes_above_ma200(m15: pd.DataFrame, signal_ts: pd.Timestamp) -> int:
    """當根必須收在 MA200 上；已走完的十五分K可用收盤站上（含剛好碰到）。"""
    last = m15.iloc[-1]
    if last["close"] <= last["ma200"] or pd.isna(last["ma200"]):
        return 0
    completed = 0
    for i in range(len(m15) - 2, -1, -1):
        row = m15.iloc[i]
        if pd.isna(row["ma200"]) or row["close"] < row["ma200"]:
            break
        completed += 1
    last_ts = pd.Timestamp(m15.index[-1])
    mark = pd.Timestamp(signal_ts)
    if last_ts.tzinfo is not None:
        mark = mark.tz_convert(last_ts.tzinfo) if mark.tzinfo else mark.tz_localize(last_ts.tzinfo)
    elif mark.tzinfo is not None:
        mark = mark.tz_localize(None)
    elapsed = int((mark - last_ts).total_seconds() // 60) + 5
    elapsed = min(M15_BAR_MINUTES, max(5, elapsed))
    return completed * M15_BAR_MINUTES + elapsed


def _wanted_sides(side: str) -> tuple[str, ...]:
    if side == "both":
        return ("long", "short")
    if side in ("long", "short"):
        return (side,)
    raise ValueError(f"unknown side: {side}")


def _passes(snap: AlertSnapshot, side: str) -> bool:
    if side == "short":
        return snap.ribbon_down and snap.crossed_below_ma200 and snap.close_below_all_mas
    return snap.ribbon_fanned and snap.crossed_above_ma200 and snap.close_above_all_mas


def iter_5m_ma200_alerts(
    df: pd.DataFrame,
    *,
    since: pd.Timestamp | None = None,
    until: pd.Timestamp | None = None,
    side: str = "long",
) -> list[AlertSnapshot]:
    """同一交易日連續五分 K。多方：MA5>MA10>MA20 且往上，剛站上 MA200；空方鏡像跌破。含開盤第一根。"""
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
    wanted = _wanted_sides(side)
    for i in range(start, len(work)):
        ts = work.index[i]
        if until is not None and ts > until:
            break
        snap = _snapshot_at(work, i)
        if snap is None:
            continue
        for want in wanted:
            if _passes(snap, want):
                hits.append(replace(snap, side=want))
                break
    return hits


def iter_15m_ma200_alerts(
    df: pd.DataFrame,
    *,
    since: pd.Timestamp | None = None,
    until: pd.Timestamp | None = None,
    side: str = "long",
) -> list[AlertSnapshot]:
    """同一交易日連續十五分 K。多方剛站上／空方剛跌破十五分 MA200（含開盤第一根）。"""
    if df is None or df.empty or "close" not in df.columns:
        return []
    m15 = resample_ohlcv(df, "15min")
    if len(m15) < MA_LONG + 1:
        return []
    work = add_moving_averages(m15)
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
    wanted = _wanted_sides(side)
    for i in range(start, len(work)):
        ts = work.index[i]
        if until is not None and ts > until:
            break
        snap = _snapshot_at(work, i)
        if snap is None:
            continue
        matched_side = next((want for want in wanted if _passes(snap, want)), None)
        if matched_side is None:
            continue
        bar_end = pd.Timestamp(ts) + pd.Timedelta(minutes=15)
        five_window = df[df.index < bar_end]
        if five_window.empty:
            continue
        signal_ts = pd.Timestamp(five_window.index[-1])
        hits.append(replace(snap, timestamp=signal_ts, side=matched_side))
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

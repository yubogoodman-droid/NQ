"""五分 K：MA5/10/20 多頭發散，當根收盤站上所有均線，且當下與前一根小時K都在小時 MA200 之上。"""

from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd

from tw.kline import resample_ohlcv


MA_FAST = 5
MA_MID = 10
MA_SLOW = 20
MA_LONG = 200
H1_MA = 20
H1_MA_LONG = 200
# MA5 相對 MA20 至少拉開這麼多（％），否則算糾結。
MIN_RIBBON_FAN_PCT = 0.50
# MA5–MA10、MA10–MA20 各自相對收盤至少這麼多（％），避免其中兩條黏在一起。
MIN_RIBBON_GAP_PCT = 0.15


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
    h1_ma200: float | None = None
    h1_prev_close: float | None = None
    h1_prev_ma200: float | None = None

    @property
    def bullish_aligned(self) -> bool:
        return self.ma5 > self.ma10 > self.ma20

    @property
    def mas_rising(self) -> bool:
        return self.ma5 > self.prev_ma5 and self.ma10 > self.prev_ma10 and self.ma20 > self.prev_ma20

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
        """多頭排列且均線發散：三條往上打開，不是黏在一起。"""
        return (
            self.bullish_aligned
            and self.mas_rising
            and self.ribbon_fan_pct >= MIN_RIBBON_FAN_PCT
            and self.gap_5_10_pct >= MIN_RIBBON_GAP_PCT
            and self.gap_10_20_pct >= MIN_RIBBON_GAP_PCT
        )

    @property
    def crossed_above_ma200(self) -> bool:
        return self.close > self.ma200 and self.prev_close <= self.prev_ma200

    @property
    def close_above_all_mas(self) -> bool:
        return (
            self.close > self.ma5
            and self.close > self.ma10
            and self.close > self.ma20
            and self.close > self.ma200
        )

    @property
    def hourly_close_above_ma20(self) -> bool:
        return (
            self.h1_close is not None
            and self.h1_ma20 is not None
            and self.h1_close > self.h1_ma20
        )

    @property
    def hourly_two_bars_above_ma200(self) -> bool:
        return (
            self.h1_close is not None
            and self.h1_ma200 is not None
            and self.h1_prev_close is not None
            and self.h1_prev_ma200 is not None
            and self.h1_close > self.h1_ma200
            and self.h1_prev_close > self.h1_prev_ma200
        )


@dataclass(frozen=True)
class HourlyContext:
    close: float
    ma20: float
    ma200: float
    prev_close: float
    prev_ma200: float


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["close"]
    out["ma5"] = close.rolling(MA_FAST, min_periods=MA_FAST).mean()
    out["ma10"] = close.rolling(MA_MID, min_periods=MA_MID).mean()
    out["ma20"] = close.rolling(MA_SLOW, min_periods=MA_SLOW).mean()
    out["ma200"] = close.rolling(MA_LONG, min_periods=MA_LONG).mean()
    return out


def _hourly_upto(
    five_min: pd.DataFrame,
    hourly_full: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """已收盤的小時K用獨立資料，當下這根小時K只看到訊號當下的五分K，避免偷看未來。"""
    if five_min is None or five_min.empty:
        return resample_ohlcv(five_min, "1h")
    if hourly_full is None or hourly_full.empty:
        return resample_ohlcv(five_min, "1h")
    end = pd.Timestamp(five_min.index[-1])
    hour_start = end.floor("h")
    completed = hourly_full.copy()
    idx = pd.DatetimeIndex(completed.index)
    if idx.tz is None:
        idx = idx.tz_localize(end.tz if end.tz is not None else "Asia/Taipei")
    elif end.tz is not None:
        idx = idx.tz_convert(end.tz)
    completed.index = idx
    completed = completed[completed.index < hour_start]
    forming = resample_ohlcv(five_min[five_min.index >= hour_start], "1h")
    if forming.empty:
        return completed
    if completed.empty:
        return forming
    out = pd.concat([completed, forming])
    return out[~out.index.duplicated(keep="last")].sort_index()


def hourly_context(
    five_min: pd.DataFrame,
    hourly_full: pd.DataFrame | None = None,
) -> HourlyContext | None:
    """當下與前一根小時K的收盤、MA20、MA200。優先用獨立小時K，不足再由五分合成。"""
    hourly = _hourly_upto(five_min, hourly_full)
    if hourly is None or len(hourly) < H1_MA_LONG + 1 or "close" not in hourly.columns:
        return None
    close = hourly["close"]
    ma20 = close.rolling(H1_MA, min_periods=H1_MA).mean()
    ma200 = close.rolling(H1_MA_LONG, min_periods=H1_MA_LONG).mean()
    last = hourly.iloc[-1]
    prev = hourly.iloc[-2]
    if any(pd.isna(v) for v in (last["close"], prev["close"], ma20.iloc[-1], ma200.iloc[-1], ma200.iloc[-2])):
        return None
    return HourlyContext(
        close=float(last["close"]),
        ma20=float(ma20.iloc[-1]),
        ma200=float(ma200.iloc[-1]),
        prev_close=float(prev["close"]),
        prev_ma200=float(ma200.iloc[-2]),
    )


def hourly_close_and_ma20(five_min: pd.DataFrame) -> tuple[float, float] | None:
    ctx = hourly_context(five_min)
    if ctx is None:
        return None
    return ctx.close, ctx.ma20


def iter_5m_ma200_alerts(
    df: pd.DataFrame,
    *,
    since: pd.Timestamp | None = None,
    until: pd.Timestamp | None = None,
    hourly_full: pd.DataFrame | None = None,
) -> list[AlertSnapshot]:
    """同一交易日連續五分 K：短均發散、剛站上五分 MA200，且當下與前一根小時K都在小時 MA200 之上。"""
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
        if not (snap.ribbon_fanned and snap.crossed_above_ma200 and snap.close_above_all_mas):
            continue
        hourly = hourly_context(work.iloc[: i + 1], hourly_full=hourly_full)
        if hourly is None:
            continue
        if hourly.close <= hourly.ma20:
            continue
        if hourly.close <= hourly.ma200 or hourly.prev_close <= hourly.prev_ma200:
            continue
        hits.append(
            replace(
                snap,
                h1_close=hourly.close,
                h1_ma20=hourly.ma20,
                h1_ma200=hourly.ma200,
                h1_prev_close=hourly.prev_close,
                h1_prev_ma200=hourly.prev_ma200,
            )
        )
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

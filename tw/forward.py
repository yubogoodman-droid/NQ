"""訊號進場後，同一交易日往後看固定分鐘的漲跌。"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from tw.backtest_5m import BacktestHit


HOLD_MINUTES = 60


@dataclass(frozen=True)
class HourLater:
    entry: float
    later: float
    later_ts: pd.Timestamp

    @property
    def ret_pct(self) -> float:
        if not self.entry:
            return 0.0
        return (self.later / self.entry - 1.0) * 100.0

    @property
    def win(self) -> bool:
        return self.later > self.entry

    @property
    def flat(self) -> bool:
        return self.later == self.entry


@dataclass(frozen=True)
class HourLaterStats:
    n_hits: int
    n_scored: int
    n_short: int
    wins: int
    flats: int
    losses: int
    avg_pct: float | None
    med_pct: float | None

    @property
    def win_rate(self) -> float | None:
        if self.n_scored <= 0:
            return None
        return self.wins / self.n_scored * 100.0


def hour_later(
    frame: pd.DataFrame | None,
    ts: pd.Timestamp,
    entry: float,
    minutes: int = HOLD_MINUTES,
) -> HourLater | None:
    """進場收盤 vs 同一交易日 +minutes 那根五分K收盤。尾盤不夠則 None。"""
    if frame is None or frame.empty or "close" not in frame.columns or not entry:
        return None
    work = frame.sort_index()
    mark = pd.Timestamp(ts)
    idx = work.index
    if not isinstance(idx, pd.DatetimeIndex):
        return None
    if idx.tz is not None:
        mark = mark.tz_convert(idx.tz) if mark.tzinfo else mark.tz_localize(idx.tz)
    elif mark.tzinfo is not None:
        mark = mark.tz_localize(None)
    target = mark + pd.Timedelta(minutes=minutes)
    day = mark.date()
    later = work[(idx >= target) & (pd.Series(idx.date, index=idx) == day)]
    if later.empty:
        return None
    row = later.iloc[0]
    return HourLater(entry=float(entry), later=float(row["close"]), later_ts=later.index[0])


def hour_later_for_hit(hit: BacktestHit, minutes: int = HOLD_MINUTES) -> HourLater | None:
    return hour_later(hit.frame, hit.snapshot.timestamp, hit.snapshot.close, minutes=minutes)


def summarize_hour_later(hits: list[BacktestHit], minutes: int = HOLD_MINUTES) -> HourLaterStats:
    scored: list[HourLater] = []
    short = 0
    for hit in hits:
        move = hour_later_for_hit(hit, minutes=minutes)
        if move is None:
            short += 1
        else:
            scored.append(move)
    rets = [m.ret_pct for m in scored]
    return HourLaterStats(
        n_hits=len(hits),
        n_scored=len(scored),
        n_short=short,
        wins=sum(1 for m in scored if m.win),
        flats=sum(1 for m in scored if m.flat),
        losses=sum(1 for m in scored if not m.win and not m.flat),
        avg_pct=(sum(rets) / len(rets)) if rets else None,
        med_pct=(float(pd.Series(rets).median()) if rets else None),
    )

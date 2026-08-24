"""盤中監控共用：tick 合成五分K、格式化 Telegram、掃描剛收完的那根。"""

from __future__ import annotations

import html
from dataclasses import dataclass, field
from datetime import datetime, time
from zoneinfo import ZoneInfo

import pandas as pd

from tw.signals import AlertSnapshot, iter_15m_ma240_alerts, iter_5m_ma240_alerts

TAIPEI = ZoneInfo("Asia/Taipei")
SESSION_OPEN = time(9, 0)
SESSION_CLOSE = time(13, 30)


@dataclass
class OhlcvBar:
    start: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


@dataclass
class BarAggregator:
    minutes: int = 5
    forming: OhlcvBar | None = None
    closed: list[OhlcvBar] = field(default_factory=list)

    def on_tick(self, ts: pd.Timestamp, price: float, volume: float = 0.0) -> OhlcvBar | None:
        """吃一筆成交。若跨進下一根，回傳剛收完的那根。"""
        if price <= 0:
            return None
        mark = _as_taipei(ts)
        bucket = floor_bar(mark, self.minutes)
        if self.forming is None:
            self.forming = OhlcvBar(bucket, price, price, price, price, volume)
            return None
        if bucket > self.forming.start:
            done = self.forming
            self.forming = OhlcvBar(bucket, price, price, price, price, volume)
            self.closed.append(done)
            return done
        self.forming.high = max(self.forming.high, price)
        self.forming.low = min(self.forming.low, price)
        self.forming.close = price
        self.forming.volume += volume
        return None

    def flush_if_due(self, now: pd.Timestamp) -> OhlcvBar | None:
        if self.forming is None:
            return None
        mark = _as_taipei(now)
        if mark >= self.forming.start + pd.Timedelta(minutes=self.minutes):
            done = self.forming
            self.forming = None
            self.closed.append(done)
            return done
        return None


def floor_bar(ts: pd.Timestamp, minutes: int = 5) -> pd.Timestamp:
    mark = _as_taipei(ts)
    minute = (mark.minute // minutes) * minutes
    return mark.replace(minute=minute, second=0, microsecond=0)


def _as_taipei(ts: pd.Timestamp | datetime) -> pd.Timestamp:
    mark = pd.Timestamp(ts)
    if mark.tzinfo is None:
        return mark.tz_localize(TAIPEI)
    return mark.tz_convert(TAIPEI)


def in_session(now: datetime | pd.Timestamp | None = None) -> bool:
    mark = _as_taipei(now or datetime.now(TAIPEI))
    if mark.weekday() >= 5:
        return False
    clock = mark.time()
    return SESSION_OPEN <= clock <= SESSION_CLOSE


def kbars_to_ohlcv(kbars) -> pd.DataFrame:
    """永豐 api.kbars 轉成訊號用的 OHLCV（1 分K）。"""
    if kbars is None:
        return _empty_ohlcv()
    try:
        frame = pd.DataFrame({**kbars})
    except Exception:
        return _empty_ohlcv()
    if frame.empty:
        return _empty_ohlcv()
    cols = {str(c).lower(): c for c in frame.columns}
    ts_col = cols.get("ts") or cols.get("time")
    if ts_col is None:
        return _empty_ohlcv()
    rename = {}
    for dest, aliases in (
        ("open", ("open",)),
        ("high", ("high",)),
        ("low", ("low",)),
        ("close", ("close",)),
        ("volume", ("volume",)),
    ):
        src = next((cols[a] for a in aliases if a in cols), None)
        if src is not None:
            rename[src] = dest
    out = frame.rename(columns=rename)
    out.index = pd.to_datetime(out[ts_col])
    if out.index.tz is None:
        out.index = out.index.tz_localize(TAIPEI)
    else:
        out.index = out.index.tz_convert(TAIPEI)
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in out.columns]
    return out[keep].astype(float).sort_index()


def _empty_ohlcv() -> pd.DataFrame:
    return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])


def upsert_bar(frame: pd.DataFrame, bar: OhlcvBar) -> pd.DataFrame:
    row = pd.DataFrame(
        {
            "open": [bar.open],
            "high": [bar.high],
            "low": [bar.low],
            "close": [bar.close],
            "volume": [bar.volume],
        },
        index=[_as_taipei(bar.start)],
    )
    if frame is None or frame.empty:
        return row
    work = frame.copy()
    work.loc[row.index[0], row.columns] = row.iloc[0]
    return work.sort_index()


def alerts_on_closed_bar(
    frame: pd.DataFrame,
    bar: OhlcvBar,
    *,
    tf: str,
    side: str = "long",
) -> list[AlertSnapshot]:
    """只收「剛收完這根」對應的通知，避免把歷史交叉重發。"""
    if frame is None or frame.empty:
        return []
    mark = _as_taipei(bar.start)
    day = mark.normalize()
    until = mark + pd.Timedelta(minutes=4, seconds=59)
    if tf == "15m":
        hits = iter_15m_ma240_alerts(
            frame, since=day, until=until + pd.Timedelta(minutes=10), side=side
        )
        return [h for h in hits if _same_15m(h.timestamp, mark)]
    hits = iter_5m_ma240_alerts(frame, since=day, until=until, side=side)
    return [h for h in hits if _as_taipei(h.timestamp) == mark]


def _same_15m(signal_ts: pd.Timestamp, five_start: pd.Timestamp) -> bool:
    sig = _as_taipei(signal_ts)
    left = floor_bar(five_start, 15)
    # 15 分K 在 09:10／09:25… 這根五分收完；訊號時間記最後一根五分。
    return floor_bar(sig, 15) == left and five_start.minute % 15 == 10


def format_telegram(name: str, symbol: str, snap: AlertSnapshot, tf: str) -> str:
    short = getattr(snap, "side", "long") == "short"
    if tf == "15m":
        title = "十五分K 剛跌破 MA240" if short else "十五分K 剛站上 MA240"
    else:
        title = "五分K 剛跌破 MA240" if short else "五分K 剛站上 MA240"
    cmp = "&lt;" if short else "&gt;"
    ts = pd.Timestamp(snap.timestamp).tz_convert(TAIPEI).strftime("%H:%M")
    url = f"https://tw.stock.yahoo.com/quote/{symbol}"
    lines = [
        f"<b>{title}</b>",
        f"{html.escape(name)} {html.escape(symbol)}",
        f"時間 {ts}",
        f"收盤 {snap.close:.2f} {cmp} MA240 {snap.ma240:.2f}",
        f"短均 {snap.ma5:.2f} / {snap.ma10:.2f} / {snap.ma20:.2f}",
    ]
    lines.append(url)
    return "\n".join(lines)


def should_run_15m(bar: OhlcvBar) -> bool:
    return _as_taipei(bar.start).minute % 15 == 10

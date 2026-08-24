"""K 線下載：有永豐金鑰就走 Shioaji，否則 Yahoo。"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

TAIPEI = ZoneInfo("Asia/Taipei")
# Yahoo 一分K 單次請求最多 8 個日曆天；更長區間要分段再合併。
YAHOO_1M_MAX_DAYS = 8
_source = "auto"


def set_kline_source(source: str) -> None:
    """auto / shioaji / yahoo。"""
    global _source
    allowed = {"auto", "shioaji", "yahoo"}
    if source not in allowed:
        raise ValueError(f"kline source must be one of {sorted(allowed)}")
    _source = source


def kline_source() -> str:
    return _source


def using_shioaji() -> bool:
    if _source == "yahoo":
        return False
    from tw.shioaji_feed import configured

    if _source == "shioaji":
        if not configured():
            raise RuntimeError("指定 --source shioaji，但未設定 SHIOAJI_API_KEY / SHIOAJI_SECRET_KEY")
        return True
    return configured()


def fetch_1m_bars(
    symbol: str,
    range_: str = "5d",
    closed_only: bool = False,
    start: date | str | None = None,
    end: date | str | None = None,
    **_: object,
) -> pd.DataFrame:
    """下載單一標的一分 K。"""
    frames = fetch_bars_many(
        [symbol],
        interval="1m",
        range_=range_,
        closed_only=closed_only,
        start=start,
        end=end,
    )
    return frames.get(symbol, _empty())


def fetch_1m_bars_many(
    symbols: list[str],
    range_: str = "5d",
    closed_only: bool = False,
    start: date | str | None = None,
    end: date | str | None = None,
) -> dict[str, pd.DataFrame]:
    return fetch_bars_many(
        symbols,
        interval="1m",
        range_=range_,
        closed_only=closed_only,
        start=start,
        end=end,
    )


def kline_window_for_date(on_date: date, lookback_days: int = 7) -> tuple[date, date]:
    """回測某日時，往前 lookback_days 抓 K 線（含前一交易日，MA240 才算得出來）。end 不含當天之後。"""
    return on_date - timedelta(days=lookback_days), on_date + timedelta(days=1)


def date_windows(start: date, end: date, max_days: int = YAHOO_1M_MAX_DAYS) -> list[tuple[date, date]]:
    """把 [start, end) 切成 Yahoo 一分K 能一次抓完的視窗。"""
    if end <= start:
        return []
    windows: list[tuple[date, date]] = []
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=max_days), end)
        windows.append((cur, nxt))
        cur = nxt
    return windows


def fetch_bars_many(
    symbols: list[str],
    interval: str = "1m",
    range_: str = "5d",
    closed_only: bool = False,
    start: date | str | None = None,
    end: date | str | None = None,
) -> dict[str, pd.DataFrame]:
    """一次抓多檔 K 線，回傳 symbol -> OHLCV。"""
    unique = list(dict.fromkeys(s for s in symbols if s))
    if not unique:
        return {}
    if using_shioaji():
        from tw import shioaji_feed

        try:
            return shioaji_feed.fetch_bars_many(
                unique,
                interval=interval,
                range_=range_,
                closed_only=closed_only,
                start=start,
                end=end,
            )
        except Exception:
            if kline_source() == "shioaji":
                raise
            # auto 模式登入失敗就退回 Yahoo，不要讓掃描整段死掉
            pass

    start_d = _as_date(start)
    end_d = _as_date(end)
    if start_d is not None and end_d is not None and interval == "1m":
        chunks = [
            _download_normalized(unique, interval, closed_only=closed_only, start=a, end=b)
            for a, b in date_windows(start_d, end_d)
        ]
        return _concat_symbol_frames(chunks)

    return _download_normalized(
        unique,
        interval,
        closed_only=closed_only,
        period=None if start_d is not None else range_,
        start=start_d,
        end=end_d,
    )


def _as_date(value: date | str | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _download_normalized(
    symbols: list[str],
    interval: str,
    *,
    closed_only: bool,
    period: str | None = None,
    start: date | None = None,
    end: date | None = None,
) -> dict[str, pd.DataFrame]:
    kwargs: dict = {
        "interval": interval,
        "group_by": "ticker",
        "auto_adjust": True,
        "threads": True,
        "progress": False,
    }
    if start is not None and end is not None:
        kwargs["start"] = start.isoformat()
        kwargs["end"] = end.isoformat()
    else:
        kwargs["period"] = period or "5d"
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("指定 Yahoo K 線時才需要 yfinance：pip install yfinance") from exc
    raw = yf.download(symbols, **kwargs)
    out: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        frame = _slice_ticker(raw, symbol)
        normalized = _normalize_ohlcv(frame, closed_only=closed_only, interval=interval)
        if not normalized.empty:
            out[symbol] = normalized
    return out


def _concat_symbol_frames(chunks: list[dict[str, pd.DataFrame]]) -> dict[str, pd.DataFrame]:
    keys: list[str] = []
    for chunk in chunks:
        for symbol in chunk:
            if symbol not in keys:
                keys.append(symbol)
    out: dict[str, pd.DataFrame] = {}
    for symbol in keys:
        parts = [chunk[symbol] for chunk in chunks if symbol in chunk and not chunk[symbol].empty]
        if not parts:
            continue
        merged = pd.concat(parts).sort_index()
        merged = merged[~merged.index.duplicated(keep="last")]
        out[symbol] = merged
    return out


def _slice_ticker(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return _empty()
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = set(map(str, raw.columns.get_level_values(0)))
        if symbol in level0:
            return raw[symbol].copy()
        last = set(map(str, raw.columns.get_level_values(-1)))
        if symbol in last:
            return raw.xs(symbol, axis=1, level=-1).copy()
        return _empty()
    # 單檔時 yfinance 可能不帶 ticker 層
    return raw.copy()


def _normalize_ohlcv(
    df: pd.DataFrame,
    closed_only: bool = False,
    interval: str = "1m",
) -> pd.DataFrame:
    if df is None or df.empty:
        return _empty()
    work = df.copy()
    if isinstance(work.columns, pd.MultiIndex):
        work.columns = [str(col[-1]) for col in work.columns]
    work.columns = [str(col).split()[-1].lower() for col in work.columns]
    needed = ["open", "high", "low", "close"]
    if any(col not in work.columns for col in needed):
        return _empty()
    if "volume" not in work.columns:
        work["volume"] = 0.0
    work = work[needed + ["volume"]].dropna(subset=["close"])
    if work.empty:
        return _empty()
    index = pd.DatetimeIndex(work.index)
    if index.tz is None:
        index = index.tz_localize(TAIPEI)
    else:
        index = index.tz_convert(TAIPEI)
    work.index = index
    work = work[~work.index.duplicated(keep="last")].sort_index()
    if closed_only and len(work) >= 2:
        last = pd.Timestamp(work.index[-1])
        now = pd.Timestamp(datetime.now(TAIPEI))
        freq = "5min" if interval == "5m" else "min"
        if last.floor(freq) >= now.floor(freq):
            work = work.iloc[:-1]
    return work.astype(float)


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

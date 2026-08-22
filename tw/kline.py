"""Yahoo Finance 五分 / 一分 K 下載。"""

from __future__ import annotations

import time
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd

TAIPEI = ZoneInfo("Asia/Taipei")


def resample_ohlcv(df: pd.DataFrame, rule: str = "15min") -> pd.DataFrame:
    """把較短週期 OHLCV 合成較長週期（預設五分 → 十五分）。"""
    if df is None or df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    work = df.copy()
    if not isinstance(work.index, pd.DatetimeIndex):
        work.index = pd.DatetimeIndex(work.index)
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    cols = {name: how for name, how in agg.items() if name in work.columns}
    out = work.resample(rule, label="left", closed="left").agg(cols)
    return out.dropna(subset=["close"]) if "close" in out.columns else out


def fetch_bars_many(
    symbols: list[str],
    interval: str = "5m",
    range_: str = "1mo",
    closed_only: bool = False,
    start: date | str | None = None,
    end: date | str | None = None,
    batch_size: int = 20,
    retries: int = 4,
) -> dict[str, pd.DataFrame]:
    """一次抓多檔 K 線，回傳 symbol -> OHLCV。"""
    unique = list(dict.fromkeys(s for s in symbols if s))
    if not unique:
        return {}
    start_d = _as_date(start)
    end_d = _as_date(end)
    out: dict[str, pd.DataFrame] = {}
    for i in range(0, len(unique), batch_size):
        batch = unique[i : i + batch_size]
        chunk = _download_with_retry(
            batch,
            interval,
            closed_only=closed_only,
            period=None if start_d is not None else range_,
            start=start_d,
            end=end_d,
            retries=retries,
        )
        out.update(chunk)
        if i + batch_size < len(unique):
            time.sleep(1.2)
    return out


def _as_date(value: date | str | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _download_with_retry(
    symbols: list[str],
    interval: str,
    *,
    closed_only: bool,
    period: str | None,
    start: date | None,
    end: date | None,
    retries: int,
) -> dict[str, pd.DataFrame]:
    last_error: Exception | None = None
    delay = 3.0
    for attempt in range(retries):
        try:
            return _download_normalized(
                symbols,
                interval,
                closed_only=closed_only,
                period=period,
                start=start,
                end=end,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(delay)
            delay *= 1.8
    if last_error is not None:
        print(f"K線下載失敗 {symbols[:3]}…：{last_error}", flush=True)
    return {}


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
        kwargs["period"] = period or "1mo"
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("需要 yfinance：pip install yfinance") from exc
    raw = yf.download(symbols, **kwargs)
    out: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        frame = _slice_ticker(raw, symbol)
        normalized = _normalize_ohlcv(frame, closed_only=closed_only, interval=interval)
        if not normalized.empty:
            out[symbol] = normalized
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
    return raw.copy()


def _normalize_ohlcv(
    df: pd.DataFrame,
    closed_only: bool = False,
    interval: str = "5m",
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
        if interval == "5m":
            freq = "5min"
        elif interval == "1h":
            freq = "h"
        elif interval in ("1d", "1wk"):
            freq = "D"
        else:
            freq = "min"
        if last.floor(freq) >= now.floor(freq):
            work = work.iloc[:-1]
    return work.astype(float)


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

"""Yahoo Finance 一分 K（yfinance 批次下載，避開 429）。"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

TAIPEI = ZoneInfo("Asia/Taipei")


def fetch_1m_bars(
    symbol: str,
    range_: str = "5d",
    closed_only: bool = False,
    **_: object,
) -> pd.DataFrame:
    """下載單一標的一分 K。"""
    frames = fetch_bars_many([symbol], interval="1m", range_=range_, closed_only=closed_only)
    return frames.get(symbol, _empty())


def fetch_1m_bars_many(
    symbols: list[str],
    range_: str = "5d",
    closed_only: bool = False,
) -> dict[str, pd.DataFrame]:
    return fetch_bars_many(symbols, interval="1m", range_=range_, closed_only=closed_only)


def fetch_bars_many(
    symbols: list[str],
    interval: str = "1m",
    range_: str = "5d",
    closed_only: bool = False,
) -> dict[str, pd.DataFrame]:
    """一次抓多檔 K 線，回傳 symbol -> OHLCV。"""
    unique = list(dict.fromkeys(s for s in symbols if s))
    if not unique:
        return {}

    raw = yf.download(
        unique,
        interval=interval,
        period=range_,
        group_by="ticker",
        auto_adjust=True,
        threads=True,
        progress=False,
    )
    out: dict[str, pd.DataFrame] = {}
    for symbol in unique:
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

"""從 Yahoo Finance 下載台股日線（選池）與一分 K（進場）。"""

from __future__ import annotations

from datetime import time
from pathlib import Path

import pandas as pd
import yfinance as yf

from tw.universe import TwStock

OHLCV = ("open", "high", "low", "close", "volume")
SESSION_START = time(9, 0)
SESSION_END = time(13, 30)


def _to_taipei_naive(idx: pd.Index, *, midnight: bool = False) -> pd.DatetimeIndex:
    out = pd.DatetimeIndex(idx)
    if out.tz is not None:
        out = out.tz_convert("Asia/Taipei").tz_localize(None)
    if midnight:
        out = out.normalize()
    return out


def _in_session(idx: pd.DatetimeIndex) -> pd.Series:
    clock = idx.time
    return pd.Series(
        [(SESSION_START <= t <= SESSION_END) for t in clock],
        index=idx,
    )


def _extract_ticker_frame(
    raw: pd.DataFrame,
    ticker: str,
    *,
    keep_time: bool,
) -> pd.DataFrame | None:
    if raw is None or raw.empty:
        return None

    if isinstance(raw.columns, pd.MultiIndex):
        level0 = raw.columns.get_level_values(0)
        level1 = raw.columns.get_level_values(1)
        if ticker in level0:
            part = raw[ticker].copy()
        elif ticker in level1:
            part = raw.xs(ticker, axis=1, level=1).copy()
        else:
            return None
    else:
        part = raw.copy()

    part.columns = [str(c).lower().replace(" ", "_") for c in part.columns]
    if "close" not in part.columns:
        return None
    if "volume" not in part.columns:
        part["volume"] = 0
    for col in ("open", "high", "low"):
        if col not in part.columns:
            part[col] = part["close"]
    part = part[list(OHLCV)].apply(pd.to_numeric, errors="coerce")
    part = part.dropna(subset=["close"])
    if part.empty:
        return None
    part.index = _to_taipei_naive(part.index, midnight=not keep_time)
    part = part[~part.index.duplicated(keep="last")].sort_index()
    if keep_time:
        part = part.loc[_in_session(part.index).values]
    return part if not part.empty else None


def _download_chunks(
    stocks: list[TwStock],
    *,
    label: str,
    interval: str,
    keep_time: bool,
    min_bars: int,
    chunk_size: int,
    cache_path: str | Path | None,
    period: str | None = None,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, pd.DataFrame]:
    cache = Path(cache_path) if cache_path else None
    if cache and cache.exists():
        payload = pd.read_pickle(cache)
        frames = {}
        for ticker, frame in payload.items():
            out = frame.copy()
            out.index = _to_taipei_naive(out.index, midnight=not keep_time)
            frames[str(ticker)] = out[list(OHLCV)]
        return frames

    frames: dict[str, pd.DataFrame] = {}
    tickers = [s.ticker for s in stocks]
    total = len(tickers)
    for i in range(0, total, chunk_size):
        chunk = tickers[i : i + chunk_size]
        print(f"{label} {i + 1}-{min(i + chunk_size, total)}/{total}", flush=True)
        kwargs: dict = dict(
            group_by="ticker",
            threads=True,
            auto_adjust=False,
            progress=False,
            interval=interval,
        )
        if period:
            kwargs["period"] = period
        else:
            kwargs["start"] = start
            kwargs["end"] = end
        raw = yf.download(chunk, **kwargs)
        if raw is None or raw.empty:
            continue
        for ticker in chunk:
            frame = _extract_ticker_frame(raw, ticker, keep_time=keep_time)
            if frame is not None and len(frame) >= min_bars:
                frames[ticker] = frame

    if cache and frames:
        cache.parent.mkdir(parents=True, exist_ok=True)
        pd.to_pickle(frames, cache)
    return frames


def download_daily(
    stocks: list[TwStock],
    *,
    period: str = "1mo",
    chunk_size: int = 80,
    cache_path: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """日線，用來算週成交額排名。"""
    return _download_chunks(
        stocks,
        label="下載日線",
        interval="1d",
        keep_time=False,
        min_bars=5,
        chunk_size=chunk_size,
        cache_path=cache_path,
        period=period,
    )


def download_minute(
    stocks: list[TwStock],
    *,
    period: str = "7d",
    chunk_size: int = 20,
    cache_path: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """一分 K。Yahoo 大約只提供近 7 個交易日。"""
    return _download_chunks(
        stocks,
        label="下載一分K",
        interval="1m",
        keep_time=True,
        min_bars=220,
        chunk_size=chunk_size,
        cache_path=cache_path,
        period=period,
    )


def to_panels(
    frames: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """轉成 open/high/low/close/volume 面板（columns=ticker）。"""
    opens = pd.concat({t: f["open"] for t, f in frames.items()}, axis=1).sort_index()
    highs = pd.concat({t: f["high"] for t, f in frames.items()}, axis=1).sort_index()
    lows = pd.concat({t: f["low"] for t, f in frames.items()}, axis=1).sort_index()
    closes = pd.concat({t: f["close"] for t, f in frames.items()}, axis=1).sort_index()
    volumes = pd.concat({t: f["volume"] for t, f in frames.items()}, axis=1).sort_index()
    return opens, highs, lows, closes, volumes

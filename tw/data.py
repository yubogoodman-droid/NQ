"""從 Yahoo Finance 下載台股日線，並整理成收盤 / 成交量面板。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

from tw.universe import TwStock

OHLCV = ("open", "high", "low", "close", "volume")


def _normalize_index(idx: pd.Index) -> pd.DatetimeIndex:
    out = pd.DatetimeIndex(idx)
    if out.tz is not None:
        out = out.tz_convert("Asia/Taipei").tz_localize(None)
    return out.normalize()


def _extract_ticker_frame(raw: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
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
    part.index = _normalize_index(part.index)
    part = part[~part.index.duplicated(keep="last")].sort_index()
    return part


def download_ohlcv(
    stocks: list[TwStock],
    *,
    start: str,
    end: str | None = None,
    chunk_size: int = 80,
    cache_path: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """下載日線。回傳 {ticker: OHLCV DataFrame}。可寫入 parquet 快取。"""
    cache = Path(cache_path) if cache_path else None
    if cache and cache.exists():
        payload = pd.read_pickle(cache)
        frames = {}
        for ticker, frame in payload.items():
            out = frame.copy()
            out.index = pd.DatetimeIndex(out.index).normalize()
            frames[str(ticker)] = out[list(OHLCV)]
        return frames

    frames: dict[str, pd.DataFrame] = {}
    tickers = [s.ticker for s in stocks]
    total = len(tickers)
    for i in range(0, total, chunk_size):
        chunk = tickers[i : i + chunk_size]
        print(f"下載日線 {i + 1}-{min(i + chunk_size, total)}/{total}", flush=True)
        raw = yf.download(
            chunk,
            start=start,
            end=end,
            group_by="ticker",
            threads=True,
            auto_adjust=False,
            progress=False,
        )
        if raw is None or raw.empty:
            continue
        for ticker in chunk:
            frame = _extract_ticker_frame(raw, ticker)
            if frame is not None and len(frame) >= 220:
                frames[ticker] = frame

    if cache and frames:
        cache.parent.mkdir(parents=True, exist_ok=True)
        pd.to_pickle(frames, cache)

    return frames


def to_panels(frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """轉成 open/high/low/close/volume 面板（columns=ticker）。"""
    opens = pd.concat({t: f["open"] for t, f in frames.items()}, axis=1).sort_index()
    highs = pd.concat({t: f["high"] for t, f in frames.items()}, axis=1).sort_index()
    lows = pd.concat({t: f["low"] for t, f in frames.items()}, axis=1).sort_index()
    closes = pd.concat({t: f["close"] for t, f in frames.items()}, axis=1).sort_index()
    volumes = pd.concat({t: f["volume"] for t, f in frames.items()}, axis=1).sort_index()
    return opens, highs, lows, closes, volumes

"""Yahoo 台股一分 K。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import yfinance as yf


def fetch_1m(symbol: str, period: str = "7d") -> pd.DataFrame:
    raw = yf.Ticker(symbol).history(period=period, interval="1m", auto_adjust=False)
    if raw is None or raw.empty:
        return pd.DataFrame()
    df = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert("Asia/Taipei")
    minutes = df.index.hour * 60 + df.index.minute
    return df[(minutes >= 9 * 60) & (minutes <= 13 * 60 + 30)]


def fetch_1m_many(symbols: list[str], period: str = "7d", workers: int = 8) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(fetch_1m, sym, period): sym for sym in symbols}
        for fut in as_completed(futs):
            sym = futs[fut]
            try:
                df = fut.result()
            except Exception as exc:
                print(f"略過 {sym}: {exc}", flush=True)
                continue
            if len(df) >= 220:
                frames[sym] = df
            print(f"  {sym} {len(df)} 根", flush=True)
    return frames

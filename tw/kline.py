"""Yahoo Finance 一分 K。"""

from __future__ import annotations

import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from tw.ranking import DEFAULT_HEADERS

TAIPEI = ZoneInfo("Asia/Taipei")
CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"


def fetch_1m_bars(
    symbol: str,
    range_: str = "5d",
    session: requests.Session | None = None,
    timeout: int = 20,
    closed_only: bool = False,
) -> pd.DataFrame:
    """下載一分 K，index 為 Asia/Taipei 時區。"""
    sess = session or requests.Session()
    payload = None
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            resp = sess.get(
                CHART_URL.format(symbol=symbol),
                params={"interval": "1m", "range": range_, "includePrePost": "false"},
                headers=DEFAULT_HEADERS,
                timeout=timeout,
            )
            if resp.status_code in {429, 500, 502, 503}:
                raise requests.HTTPError(f"HTTP {resp.status_code}", response=resp)
            resp.raise_for_status()
            payload = resp.json()
            break
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            time.sleep(0.6 * (attempt + 1))
    if payload is None:
        raise RuntimeError(f"{symbol} 一分 K 下載失敗：{last_error}") from last_error
    error = (payload.get("chart") or {}).get("error")
    if error:
        raise RuntimeError(f"{symbol} 一分 K 錯誤：{error}")
    results = (payload.get("chart") or {}).get("result") or []
    if not results:
        return _empty()

    result = results[0]
    timestamps = result.get("timestamp") or []
    quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]

    rows = []
    for i, ts in enumerate(timestamps):
        close = _at(quote.get("close"), i)
        if close is None:
            continue
        rows.append(
            {
                "datetime": datetime.fromtimestamp(int(ts), tz=TAIPEI),
                "open": _at(quote.get("open"), i, close),
                "high": _at(quote.get("high"), i, close),
                "low": _at(quote.get("low"), i, close),
                "close": float(close),
                "volume": _at(quote.get("volume"), i, 0.0) or 0.0,
            }
        )

    if not rows:
        return _empty()

    df = pd.DataFrame(rows).set_index("datetime").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    if closed_only and len(df) >= 2:
        last = pd.Timestamp(df.index[-1])
        now = pd.Timestamp(datetime.now(TAIPEI))
        if last.floor("min") >= now.floor("min"):
            df = df.iloc[:-1]
    return df


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])


def _at(values: list | None, index: int, default: float | None = None) -> float | None:
    if not values or index >= len(values):
        return default
    value = values[index]
    if value is None:
        return default
    return float(value)

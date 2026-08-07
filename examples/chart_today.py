#!/usr/bin/env python3
"""抓取今日 NQ 五分 K 並產生 HTML 圖表。"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nq.chart import save_html_chart


def fetch_nq_5m(symbol: str = "NQ=F") -> "pd.DataFrame":
    import pandas as pd

    raw = yf.Ticker(symbol).history(period="1d", interval="5m")
    if raw.empty:
        raise RuntimeError(f"無法取得 {symbol} 五分 K 資料")

    df = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].copy()
    df.index = df.index.tz_convert("America/New_York")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="產生 NQ W底 HTML 圖表")
    parser.add_argument("--output", "-o", default="output/nq_w_bottom_today.html")
    parser.add_argument("--symbol", default="NQ=F")
    args = parser.parse_args()

    df = fetch_nq_5m(args.symbol)
    today = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    title = f"NQ 五分K W底進場 — {today} ({args.symbol})"

    out = save_html_chart(df, args.output, title=title)
    print(f"已產生圖表: {out.resolve()}")
    print(f"K 線: {len(df)} 根 | {df.index[0].strftime('%H:%M')} ~ {df.index[-1].strftime('%H:%M')} ET")


if __name__ == "__main__":
    main()

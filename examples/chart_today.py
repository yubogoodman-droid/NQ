#!/usr/bin/env python3
"""抓取今日 NQ 五分 K 並產生 HTML 圖表 / 交易卡片報告。"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nq.chart import save_html_chart
from nq.report import save_report_html


def fetch_nq_5m(symbol: str = "NQ=F") -> "pd.DataFrame":
    import pandas as pd

    raw = yf.Ticker(symbol).history(period="1d", interval="5m")
    if raw.empty:
        raise RuntimeError(f"無法取得 {symbol} 五分 K 資料")

    df = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].copy()
    df.index = df.index.tz_convert("America/New_York")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="產生 NQ W底 HTML")
    parser.add_argument("--output", "-o", help="輸出路徑")
    parser.add_argument("--symbol", default="NQ=F")
    parser.add_argument("--pages", action="store_true", help="輸出到 docs/index.html（GitHub Pages）")
    parser.add_argument(
        "--report",
        action="store_true",
        help="交易卡片報告（每筆分開，手機版）",
    )
    args = parser.parse_args()

    df = fetch_nq_5m(args.symbol)
    today = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")

    if args.report or args.pages:
        output = args.output or ("docs/index.html" if args.pages else "output/nq_report.html")
        title = f"NQ W底回測 — {today}"
        out = save_report_html(df, output, title=title, symbol=args.symbol)
    else:
        output = args.output or "output/nq_w_bottom_today.html"
        title = f"NQ 五分K W底進場 — {today} ({args.symbol})"
        out = save_html_chart(df, output, title=title)

    print(f"已產生: {out.resolve()}")
    print(f"K 線: {len(df)} 根 | {df.index[0].strftime('%H:%M')} ~ {df.index[-1].strftime('%H:%M')} ET")


if __name__ == "__main__":
    main()

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


def fetch_nq_5m(symbol: str = "NQ=F", *, period: str = "1d", days: int | None = None) -> "pd.DataFrame":
    import pandas as pd

    if days is not None:
        period = f"{days}d"
    raw = yf.Ticker(symbol).history(period=period, interval="5m")
    if raw.empty:
        raise RuntimeError(f"無法取得 {symbol} 五分 K 資料")

    df = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].copy()
    df.index = df.index.tz_convert("America/New_York")
    return df


def today_mask(df: "pd.DataFrame") -> "pd.Series":
    import pandas as pd

    today = pd.Timestamp.now(tz="America/New_York").date()
    return df.index.date == today


def main() -> None:
    parser = argparse.ArgumentParser(description="產生 NQ 破底W底 HTML")
    parser.add_argument("--output", "-o", help="輸出路徑")
    parser.add_argument("--symbol", default="NQ=F")
    parser.add_argument("--pages", action="store_true", help="輸出到 docs/index.html（GitHub Pages）")
    parser.add_argument(
        "--report",
        action="store_true",
        help="交易卡片報告（每筆分開，手機版）",
    )
    parser.add_argument("--days", type=int, help="回測天數（例如 30）")
    args = parser.parse_args()

    days = args.days
    if days:
        period = f"{days}d"
    else:
        period = "5d" if (args.report or args.pages) else "1d"

    df = fetch_nq_5m(args.symbol, period=period, days=days)
    today = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")

    if args.report or args.pages:
        output = args.output or (
            "docs/index.html" if args.pages else (f"output/nq_report_{days}d.html" if days else "output/nq_report.html")
        )
        if days:
            title = f"NQ 破底W · L2夠深 · L1L2L3 兩小時內 — 近 {days} 天"
        else:
            title = f"NQ 破底W底回測 — {today}"
        out = save_report_html(
            df, output, title=title, symbol=args.symbol, today_only=days is None
        )
    else:
        output = args.output or "output/nq_w_bottom_today.html"
        title = f"NQ 五分K 破底W底進場 — {today} ({args.symbol})"
        out = save_html_chart(df, output, title=title)

    print(f"已產生: {out.resolve()}")
    start = df.index[0].strftime("%Y-%m-%d %H:%M")
    end = df.index[-1].strftime("%Y-%m-%d %H:%M")
    print(f"K 線: {len(df)} 根 | {start} ~ {end} ET")


if __name__ == "__main__":
    main()

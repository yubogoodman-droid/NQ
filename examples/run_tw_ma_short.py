#!/usr/bin/env python3
"""台股 5/10/20 空頭排列、跌破 MA200 做空回測。"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw.backtest import run_backtest, summarize
from tw.data import download_ohlcv
from tw.report import save_report_html
from tw.strategy import TwMaShortStrategy
from tw.universe import TwStock, fetch_stock_universe


def make_demo_frames() -> tuple[dict[str, pd.DataFrame], list[TwStock]]:
    """合成一檔會觸發跌破 MA200 的日線，供無網路示範。"""
    n = 260
    idx = pd.bdate_range("2025-01-02", periods=n)
    close = np.full(n, 100.0)
    close[200:] = np.linspace(99.5, 82.0, n - 200)
    close[200] = 99.2
    high = close + 0.8
    low = close - 0.8
    open_ = np.r_[close[0], close[:-1]]
    volume = np.full(n, 5_000_000.0)
    volume[180:210] = 20_000_000.0
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )
    stock = TwStock(code="9999", name="示範股", market="TW")
    return {stock.ticker: df}, [stock]


def main() -> None:
    parser = argparse.ArgumentParser(description="台股 5/10/20 空頭排列跌破 MA200 做空回測")
    parser.add_argument("--demo", action="store_true", help="使用合成資料")
    parser.add_argument("--start", default="2022-01-01", help="下載起始日（含 MA200 暖機）")
    parser.add_argument("--bt-start", default="2023-01-01", help="回測起始日")
    parser.add_argument("--end", default=None, help="結束日")
    parser.add_argument("--top", type=int, default=100, help="週成交額前 N")
    parser.add_argument("--max-price", type=float, default=600.0)
    parser.add_argument("--max-tickers", type=int, default=0, help="除錯用：只下載前 N 檔")
    parser.add_argument("--cache", default="data/tw_ohlcv.pkl")
    parser.add_argument("--output", "-o", default="output/tw_ma_short.html")
    parser.add_argument("--pages", action="store_true", help="同時寫入 docs/tw-ma-short/index.html")
    args = parser.parse_args()

    if args.demo:
        frames, stocks = make_demo_frames()
        print("使用合成示範資料\n")
    else:
        print("抓取上市+上櫃公司清單（排除 ETF）...", flush=True)
        stocks = fetch_stock_universe()
        if args.max_tickers:
            stocks = stocks[: args.max_tickers]
        print(f"標的 {len(stocks)} 檔，開始下載日線", flush=True)
        frames = download_ohlcv(
            stocks,
            start=args.start,
            end=args.end,
            cache_path=args.cache,
        )
        print(f"有效日線 {len(frames)} 檔", flush=True)

    strategy = TwMaShortStrategy(max_price=args.max_price)
    results = run_backtest(
        frames,
        stocks,
        strategy=strategy,
        top_n=args.top,
        max_price=args.max_price,
        start=None if args.demo else args.bt_start,
    )
    stats = summarize(results)
    print("\n=== 回測摘要 ===")
    for k, v in stats.items():
        if isinstance(v, float):
            if "pct" in k or k == "win_rate":
                print(f"{k}: {v * 100:.2f}%")
            else:
                print(f"{k}: {v:.2f}")
        else:
            print(f"{k}: {v}")

    today = datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y-%m-%d")
    title = "台股 5/10/20 空頭排列 · 跌破 MA200 做空"
    subtitle = (
        f"{today} · 週成交額前 {args.top} · 排除 ETF 與股價 > {args.max_price:.0f} · "
        f"{'示範資料' if args.demo else args.bt_start + ' 起'}"
    )
    out = save_report_html(results, frames, args.output, title=title, subtitle=subtitle, stocks=stocks)
    print(f"\n已產生: {out.resolve()}")
    if args.pages:
        pages = save_report_html(
            results,
            frames,
            "docs/tw-ma-short/index.html",
            title=title,
            subtitle=subtitle,
            stocks=stocks,
        )
        print(f"Pages: {pages.resolve()}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""台股一分 K：5/10/20 空頭排列、跌破 MA200 做空回測。"""

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
from tw.data import download_daily, download_minute, to_panels
from tw.report import save_report_html
from tw.strategy import TwMaShortStrategy
from tw.universe import (
    TwStock,
    fetch_stock_universe,
    latest_weekly_top,
    tickers_ever_eligible,
    weekly_top_n_mask,
)


def make_demo_frames() -> tuple[dict[str, pd.DataFrame], list[TwStock], pd.DataFrame | None]:
    """合成一分 K，會在 MA200 跌破時做空。"""
    n = 260
    idx = pd.date_range("2026-08-17 09:00", periods=n, freq="1min")
    close = np.full(n, 100.0)
    close[200:] = np.linspace(99.4, 96.5, n - 200)
    close[200] = 99.2
    high = close + 0.15
    low = close - 0.15
    open_ = np.r_[close[0], close[:-1]]
    volume = np.full(n, 50_000.0)
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )
    stock = TwStock(code="9999", name="示範股", market="TW")
    return {stock.ticker: df}, [stock], None


def main() -> None:
    parser = argparse.ArgumentParser(description="台股一分K 5/10/20 空頭排列跌破 MA200 做空回測")
    parser.add_argument("--demo", action="store_true", help="使用合成資料")
    parser.add_argument("--top", type=int, default=100, help="週成交額前 N")
    parser.add_argument("--max-price", type=float, default=600.0)
    parser.add_argument("--max-tickers", type=int, default=0, help="除錯用：只下載前 N 檔日線")
    parser.add_argument("--minute-period", default="7d", help="一分 K 期間（Yahoo 約最多 7d）")
    parser.add_argument("--daily-cache", default="data/tw_daily.pkl")
    parser.add_argument("--minute-cache", default="data/tw_1m.pkl")
    parser.add_argument("--output", "-o", default="output/tw_ma_short.html")
    parser.add_argument("--pages", action="store_true", help="同時寫入 docs/tw-ma-short/index.html")
    args = parser.parse_args()

    eligible_daily = None
    universe_top = None

    if args.demo:
        frames, stocks, universe_top = make_demo_frames()
        print("使用合成一分 K 示範資料\n")
    else:
        print("抓取上市+上櫃公司清單（排除 ETF）...", flush=True)
        stocks = fetch_stock_universe()
        if args.max_tickers:
            stocks = stocks[: args.max_tickers]
        print(f"標的 {len(stocks)} 檔，先用日線算上一週成交額前 {args.top}", flush=True)
        daily = download_daily(stocks, period="1mo", cache_path=args.daily_cache)
        _o, _h, _l, closes, volumes = to_panels(daily)
        eligible_daily = weekly_top_n_mask(
            closes, volumes, top_n=args.top, max_price=args.max_price
        )
        universe_top = latest_weekly_top(
            closes, volumes, stocks, top_n=args.top, max_price=args.max_price
        )
        wanted = set(tickers_ever_eligible(eligible_daily))
        minute_stocks = [s for s in stocks if s.ticker in wanted]
        print(f"週成交額條件命中 {len(minute_stocks)} 檔，下載一分 K", flush=True)
        frames = download_minute(
            minute_stocks,
            period=args.minute_period,
            cache_path=args.minute_cache,
        )
        print(f"有效一分 K {len(frames)} 檔", flush=True)
        stocks = minute_stocks

    strategy = TwMaShortStrategy(max_price=args.max_price)
    results = run_backtest(
        frames,
        stocks,
        strategy=strategy,
        eligible_daily=eligible_daily,
        max_price=args.max_price,
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
    title = "台股一分K 5/10/20 空頭排列 · 跌破 MA200 做空"
    subtitle = (
        f"{today} · 週成交額前 {args.top} · 排除 ETF 與股價 > {args.max_price:.0f} · "
        f"{'示範資料' if args.demo else 'Yahoo 近 ' + args.minute_period + ' 一分K'}"
    )
    out = save_report_html(
        results,
        frames,
        args.output,
        title=title,
        subtitle=subtitle,
        stocks=stocks,
        universe_top=universe_top,
    )
    print(f"\n已產生: {out.resolve()}")
    if args.pages:
        pages = save_report_html(
            results,
            frames,
            "docs/tw-ma-short/index.html",
            title=title,
            subtitle=subtitle,
            stocks=stocks,
            universe_top=universe_top,
        )
        print(f"Pages: {pages.resolve()}")


if __name__ == "__main__":
    main()

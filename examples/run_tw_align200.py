#!/usr/bin/env python3
"""一分K：5>10>20>60 多頭排列且站上 MA200。回測近一週每日成交額前 100。"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nq.align200 import detect_align200, run_align200_backtest, summarize_align
from nq.align200_site import write_report
from nq.yahoo_1m import fetch_1m_many
from tw.ranking import RankedStock, fetch_daily_turnover_ranking, filter_by_price, filter_etfs, iter_recent_sessions


def last_weekdays(end: date | None = None, n: int = 5) -> list[date]:
    end = end or date.today()
    days = list(iter_recent_sessions(end if end.weekday() < 5 else end - timedelta(days=1), limit=n))
    return list(reversed(days))


def main() -> None:
    parser = argparse.ArgumentParser(description="台股一分K 5>10>20>60 站上MA200 週回測")
    parser.add_argument("--top", type=int, default=100)
    parser.add_argument("--days", type=int, default=5)
    parser.add_argument("--end", default="", help="結束日 YYYY-MM-DD，預設最近交易日")
    parser.add_argument("--keep-etf", action="store_true")
    parser.add_argument("--max-price", type=float, default=600.0, help="股價達此以上不掃，0=不限")
    parser.add_argument("--output", default="docs/align200")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    end = date.fromisoformat(args.end) if args.end else date.today()
    days = last_weekdays(end, args.days)
    print(f"回測日：{' '.join(d.isoformat() for d in days)}", flush=True)

    day_universe: dict[date, list[RankedStock]] = {}
    names: dict[str, str] = {}
    for day in days:
        try:
            stocks, label = fetch_daily_turnover_ranking(day, top=max(args.top * 3, 250))
        except Exception as exc:
            print(f"{day} 排行失敗：{exc}", flush=True)
            continue
        if not args.keep_etf:
            stocks = filter_etfs(stocks)
        if args.max_price > 0:
            stocks = filter_by_price(stocks, args.max_price)
        stocks = stocks[: args.top]
        day_universe[day] = stocks
        for s in stocks:
            names[s.symbol] = s.name
        print(f"{label} → 掃描 {len(stocks)} 檔（股價<{args.max_price:g}）", flush=True)

    symbols = sorted({s.symbol for stocks in day_universe.values() for s in stocks})
    print(f"下載一分K {len(symbols)} 檔…", flush=True)
    frames = fetch_1m_many(symbols, workers=args.workers)

    all_trades = []
    for day, stocks in day_universe.items():
        for stock in stocks:
            df = frames.get(stock.symbol)
            if df is None:
                continue
            sigs = detect_align200(df, symbol=stock.symbol, name=stock.name, on_date=day)
            trades = run_align200_backtest(df, sigs)
            all_trades.extend(trades)

    all_trades.sort(key=lambda t: (t.signal.day, t.signal.timestamp, t.signal.symbol))
    stats = summarize_align(all_trades)
    print(
        f"\n合計 {stats['trades']} 筆  勝率 {stats['win_rate']*100:.0f}%  "
        f"淨 {stats['total_pnl_pct_net']*100:+.2f}%  期望 {stats['expectancy_net']*100:+.3f}%",
        flush=True,
    )
    by_day: dict[date, int] = {}
    for t in all_trades:
        by_day[t.signal.day] = by_day.get(t.signal.day, 0) + 1
    for day in days:
        print(f"  {day}  {by_day.get(day, 0)} 筆", flush=True)

    notes = [
        "一分K：MA5>MA10>MA20>MA60 多頭排列，且收盤站上 MA200。",
        "只在條件剛成立那一根通知／進場（前一根尚未同時滿足）。",
        "每日成交額前 100（上市+上櫃），去掉 ETF 與股價 600 以上。09:10 前不進，13:20 平倉。",
        "MA5/10/20/60/200 全部是一分K均線（MA200＝近 200 根 1m，約 3 小時 20 分，不是日線200）。",
        "出場：收盤跌破 MA200，或 MA5 跌破 MA10。下一根開盤進場，單邊 8bps。",
        "Yahoo 一分K 約 7 日。學習用，不是下單建議。",
    ]
    out = write_report(
        Path(args.output),
        title="一分K 5>10>20>60 站上MA200 · 近一週前100",
        trades=all_trades,
        frames=frames,
        notes=notes,
    )
    print(f"\n報告：{out.resolve()}", flush=True)


if __name__ == "__main__":
    main()

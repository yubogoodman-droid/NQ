#!/usr/bin/env python3
"""台股五分 K 回測：MA5/10/20 多頭發散，當根收盤站上所有均線。"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw.backtest_5m import BacktestConfig, run_5m_backtest
from tw.report import save_backtest_html, weekday_zh


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="台股五分K多頭發散＋收盤站上所有均線，回測近 N 個交易日")
    p.add_argument("--days", type=int, default=5, help="回測交易日數（預設 5）")
    p.add_argument("--top", type=int, default=100, help="成交額前 N 名（預設 100）")
    p.add_argument("--max-price", type=float, default=650.0, help="濾掉此價格以上（預設 650）")
    p.add_argument("--include-etf", action="store_true", help="不過濾 ETF")
    p.add_argument("--include-financial", action="store_true", help="不過濾金融股")
    p.add_argument("--include-telecom", action="store_true", help="不過濾電信股")
    p.add_argument("--today", help="回測截止日 YYYY-MM-DD（預設台北今天）")
    p.add_argument("-o", "--output", default="docs/tw/backtest-5m-15m.html")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    today = date.fromisoformat(args.today) if args.today else None
    result = run_5m_backtest(
        BacktestConfig(
            days=args.days,
            top=args.top,
            max_price=args.max_price,
            exclude_etf=not args.include_etf,
            exclude_financial=not args.include_financial,
            exclude_telecom=not args.include_telecom,
            today=today,
        )
    )
    print()
    print(f"回測 {result.days[0].isoformat()}～{result.days[-1].isoformat()}  共 {len(result.hits)} 則通知")
    for day in result.days:
        hits = result.hits_on(day)
        names = "、".join(
            f"{h.stock.name} {h.snapshot.timestamp.strftime('%H:%M')}" for h in hits
        ) or "—"
        print(f"  {weekday_zh(day)} {day.isoformat()}  {len(hits)} 則  {names}")
    path = save_backtest_html(result, args.output)
    print(f"報告：{path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

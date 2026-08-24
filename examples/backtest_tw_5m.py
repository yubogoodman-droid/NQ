#!/usr/bin/env python3
"""台股五分 K 回測：多方剛站上、或空方剛跌破五分 MA200。"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw.backtest_5m import BacktestConfig, run_5m_backtest
from tw.forward import summarize_hour_later
from tw.report import save_backtest_html, weekday_zh


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="台股五分K：MA5>MA10>MA20且往上、剛站上五分MA200，回測近 N 個交易日"
    )
    p.add_argument("--days", type=int, default=5, help="回測交易日數（預設 5）")
    p.add_argument("--top", type=int, default=100, help="成交額前 N 名（預設 100）")
    p.add_argument("--max-price", type=float, default=500.0, help="濾掉此價格以上（預設 500）")
    p.add_argument("--kline-range", default="1mo", help="Yahoo 五分K區間（預設 1mo；五分K最長約 60 天，勿用 2mo／3mo）")
    p.add_argument("--include-etf", action="store_true", help="不過濾 ETF")
    p.add_argument("--include-financial", action="store_true", help="不過濾金融股")
    p.add_argument("--include-telecom", action="store_true", help="不過濾電信股")
    p.add_argument("--today", help="回測截止日 YYYY-MM-DD（預設台北今天）")
    p.add_argument(
        "--side",
        choices=("long", "short"),
        default="long",
        help="多方剛站上 MA200，或空方剛跌破（預設 long）",
    )
    p.add_argument("-o", "--output", default="docs/tw/backtest-5m-15m-1h.html")
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
            kline_range=args.kline_range,
            today=today,
            side=args.side,
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
    stats = summarize_hour_later(result.hits)
    if stats.win_rate is None:
        print("進場後一小時勝率 —（沒有滿一小時的樣本）")
    else:
        avg = f"{stats.avg_pct:+.2f}%" if stats.avg_pct is not None else "—"
        print(
            f"進場後一小時勝率 {stats.win_rate:.0f}%  "
            f"（{stats.wins} 贏 / {stats.n_scored} 則滿一小時，"
            f"{stats.flats} 平 {stats.losses} 輸 {stats.n_short} 則不足，平均 {avg}）"
        )
    path = save_backtest_html(result, args.output)
    print(f"報告：{path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

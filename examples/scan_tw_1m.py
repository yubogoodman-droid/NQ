#!/usr/bin/env python3
"""台股一分 K：MA5/10/20 多頭排列，且這根收盤剛站上 MA200。"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw.notify import format_hit_message, send_notifications
from tw.ranking import previous_friday, previous_weekdays
from tw.report import save_scan_html, save_week_index
from tw.screener import ScanConfig, hit_key, run_scan


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="台股一分K多頭排列＋剛站上MA200掃描")
    p.add_argument("--top", type=int, default=100, help="成交額前 N 名（預設 100）")
    p.add_argument("--max-price", type=float, default=650.0, help="濾掉此價格以上（預設 650）")
    p.add_argument("--watch", action="store_true", help="盤中每分鐘重掃，同一根 K 不重複通知")
    p.add_argument("--interval", type=int, default=60, help="watch 間隔秒數")
    p.add_argument("--latest-only", action="store_true", help="只看最新一根（watch 模式自動開啟）")
    p.add_argument("--closed-only", action="store_true", help="只用已收盤的一分 K（不含當根未收）")
    p.add_argument("--include-etf", action="store_true", help="不過濾 ETF")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("-o", "--output", default="docs/tw/index.html")
    p.add_argument(
        "--min-5m-gap",
        type=float,
        default=0.06,
        help="五分收盤相對五分 MA200 的最小距離（預設 0.06=6%，金居／玉晶光這類）",
    )
    p.add_argument(
        "--min-ma5-pop",
        type=float,
        default=0.01,
        help="收盤相對 MA5 的最小彈離（預設 0.01=1%，頎邦／信昌電這類）",
    )
    p.add_argument("--date", help="回測指定日（YYYY-MM-DD），只找當天金叉")
    p.add_argument("--last-friday", action="store_true", help="回測上週五")
    p.add_argument("--last-week", action="store_true", help="回測上週一到五，每天分開一頁")
    p.add_argument("--quiet-empty", action="store_true", help="沒命中時不印詳細清單")
    return p.parse_args()


def _on_date(args: argparse.Namespace) -> date | None:
    if args.last_friday:
        return previous_friday()
    if args.date:
        return date.fromisoformat(args.date)
    return None


def print_result(result, *, quiet_empty: bool) -> None:
    print(
        f"[{result.scanned_at.strftime('%H:%M:%S')}] "
        f"成交額前 {len(result.universe)}／股價濾掉 {result.price_dropped}／"
        f"ETF濾掉 {result.etf_dropped}／掃描 {len(result.candidates)}／"
        f"均線糾結濾掉 {result.tangled_dropped}／"
        f"五分未明顯站上MA200濾掉 {result.below_5m_dropped}／"
        f"命中 {len(result.hits)}／略過 {len(result.skipped)}／錯誤 {len(result.errors)}"
    )
    if result.as_of:
        print(f"  回測日期：{result.as_of.isoformat()}")
    if result.rank_time:
        print(f"  排行資料時間：{result.rank_time}")
    if not result.hits:
        if not quiet_empty:
            print("  目前沒有符合條件的標的。")
        return
    for hit in result.hits:
        s, snap = hit.stock, hit.snapshot
        chg = f"{s.change_percent:+.2f}%" if s.change_percent is not None else ""
        print(
            f"  #{s.rank:3d} {s.name:8s} {s.symbol:10s} "
            f"{s.price:8.2f} {chg:>8s}  "
            f"收 {snap.close:.2f} > MA200 {snap.ma200:.2f}  "
            f"前收 {snap.prev_close:.2f}  "
            f"MA {snap.ma5:.2f}/{snap.ma10:.2f}/{snap.ma20:.2f}  "
            f"{snap.timestamp.strftime('%H:%M')}"
        )


def scan_once(args: argparse.Namespace, seen: set) -> int:
    on_date = _on_date(args)
    output = args.output
    if on_date is not None and output == "docs/tw/index.html":
        output = f"docs/tw/backtest-{on_date.isoformat()}.html"
    result = run_scan(
        ScanConfig(
            top=args.top,
            max_price=args.max_price,
            closed_only=args.closed_only or on_date is not None,
            workers=args.workers,
            latest_only=(args.latest_only or args.watch) and on_date is None,
            exclude_etf=not args.include_etf,
            on_date=on_date,
            kline_range="7d" if on_date is not None else "5d",
            min_5m_ma200_gap=args.min_5m_gap,
            min_ma5_pop=args.min_ma5_pop,
        )
    )
    print_result(result, quiet_empty=args.quiet_empty)
    path = save_scan_html(result, output)
    print(f"  報告：{path}")

    new_hits = [h for h in result.hits if hit_key(h) not in seen]
    for h in result.hits:
        seen.add(hit_key(h))
    if new_hits and on_date is None:
        title, body = format_hit_message(new_hits)
        print()
        print(title)
        print(body)
        channels = send_notifications(title, body)
        if channels:
            print(f"  已通知：{', '.join(channels)}")
        else:
            print("  未設定 Telegram/Discord/桌面通知（仍已印在終端機）。")
            print("  可設 TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID 或 DISCORD_WEBHOOK_URL")
    if result.errors and not args.quiet_empty:
        print("  錯誤：")
        for stock, err in result.errors[:8]:
            print(f"    {stock.symbol} {err}")
        if len(result.errors) > 8:
            print(f"    …另有 {len(result.errors) - 8} 筆")
    return len(new_hits)


def scan_last_week(args: argparse.Namespace) -> int:
    days = previous_weekdays()
    results = []
    print(f"回測上週 {days[0].isoformat()}～{days[-1].isoformat()}，每天分開")
    for day in days:
        output = f"docs/tw/backtest-{day.isoformat()}.html"
        result = run_scan(
            ScanConfig(
                top=args.top,
                max_price=args.max_price,
                closed_only=True,
                workers=args.workers,
                latest_only=False,
                exclude_etf=not args.include_etf,
                on_date=day,
                kline_range="7d",
                min_5m_ma200_gap=args.min_5m_gap,
                min_ma5_pop=args.min_ma5_pop,
            )
        )
        print_result(result, quiet_empty=args.quiet_empty)
        path = save_scan_html(result, output)
        print(f"  報告：{path}")
        results.append(result)
    index = save_week_index(results, "docs/tw/week-last.md")
    print(f"  週目錄：{index}")
    print()
    print("上週分日命中：")
    for result in results:
        assert result.as_of is not None
        names = "、".join(h.stock.name for h in result.hits) or "—"
        print(f"  {result.as_of.isoformat()}  {len(result.hits)} 檔  {names}")
    return sum(len(r.hits) for r in results)


def main() -> int:
    args = parse_args()
    if args.last_week:
        scan_last_week(args)
        return 0
    seen: set = set()
    scan_once(args, seen)
    if not args.watch or _on_date(args) is not None:
        return 0
    print(f"\nwatch 模式，每 {args.interval} 秒重掃（Ctrl+C 結束）")
    try:
        while True:
            time.sleep(max(15, args.interval))
            scan_once(args, seen)
    except KeyboardInterrupt:
        print("\n已停止。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

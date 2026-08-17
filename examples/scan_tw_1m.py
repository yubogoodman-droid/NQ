#!/usr/bin/env python3
"""台股一分 K：MA5/10/20 多頭排列，且這根收盤剛站上 MA200。"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw.screener import ScanConfig, hit_key, run_scan


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="台股一分K多頭排列＋剛站上MA200掃描")
    p.add_argument("--top", type=int, default=100, help="成交額前 N 名（預設 100）")
    p.add_argument("--max-price", type=float, default=650.0, help="濾掉此價格以上（預設 650）")
    p.add_argument("--watch", action="store_true", help="盤中每分鐘重掃，同一根 K 不重複通知")
    p.add_argument("--interval", type=int, default=60, help="watch 間隔秒數")
    p.add_argument("--closed-only", action="store_true", help="只用已收盤的一分 K（不含當根未收）")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("-o", "--output", default="output/tw_1m_ma200.html")
    p.add_argument("--quiet-empty", action="store_true", help="沒命中時不印詳細清單")
    return p.parse_args()


def print_result(result, *, quiet_empty: bool) -> None:
    print(
        f"[{result.scanned_at.strftime('%H:%M:%S')}] "
        f"成交額前 {len(result.universe)}／股價<上限 {len(result.candidates)}／"
        f"命中 {len(result.hits)}／略過 {len(result.skipped)}／錯誤 {len(result.errors)}"
    )
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
    result = run_scan(
        ScanConfig(
            top=args.top,
            max_price=args.max_price,
            closed_only=args.closed_only,
            workers=args.workers,
        )
    )
    print_result(result, quiet_empty=args.quiet_empty)
    path = save_scan_html(result, args.output)
    print(f"  報告：{path}")

    new_hits = [h for h in result.hits if hit_key(h) not in seen]
    for h in result.hits:
        seen.add(hit_key(h))
    if new_hits:
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


def main() -> int:
    args = parse_args()
    seen: set = set()
    scan_once(args, seen)
    if not args.watch:
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

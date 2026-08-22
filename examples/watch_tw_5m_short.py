#!/usr/bin/env python3
"""台股五分 K 空頭：MA5 < MA10 < MA20、跌破 MA200、且 15 分／小時 K 都在 MA20 之下就推通知。

在下面填 Telegram 後執行：

    python3 examples/watch_tw_5m_short.py --test
    python3 examples/watch_tw_5m_short.py

Ctrl+C 結束。同一根 K 不會重發。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# —— 填這裡 ——
TELEGRAM_BOT_TOKEN = ""  # BotFather 給的 token
TELEGRAM_CHAT_ID = ""    # 你的 chat id


def apply_keys() -> None:
    if TELEGRAM_BOT_TOKEN.strip():
        os.environ.setdefault("TELEGRAM_BOT_TOKEN", TELEGRAM_BOT_TOKEN.strip())
    if TELEGRAM_CHAT_ID.strip():
        os.environ.setdefault("TELEGRAM_CHAT_ID", TELEGRAM_CHAT_ID.strip())


apply_keys()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw.notify import format_hit_message, send_notifications
from tw.watch import WatchConfig, hit_key, market_open, run_scan


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="台股五分K空頭排列跌破MA200且15分／小時K在MA20下即時通知")
    p.add_argument("--test", action="store_true", help="立刻掃一次並試送通知")
    p.add_argument("--interval", type=int, default=60, help="盤中重掃間隔秒數（預設 60）")
    p.add_argument("--top", type=int, default=100)
    p.add_argument("--max-price", type=float, default=650.0)
    p.add_argument("--include-etf", action="store_true")
    p.add_argument("--include-financial", action="store_true")
    p.add_argument("--include-telecom", action="store_true")
    return p.parse_args()


def _config(args: argparse.Namespace) -> WatchConfig:
    return WatchConfig(
        top=args.top,
        max_price=args.max_price,
        exclude_etf=not args.include_etf,
        exclude_financial=not args.include_financial,
        exclude_telecom=not args.include_telecom,
        latest_only=True,
    )


def print_result(result) -> None:
    print(
        f"[{result.scanned_at.strftime('%H:%M:%S')}] "
        f"成交額前 {len(result.universe)}／價{result.price_dropped}／"
        f"ETF{result.etf_dropped}／金融{result.financial_dropped}／"
        f"電信{result.telecom_dropped}／掃描 {len(result.candidates)}／"
        f"命中 {len(result.hits)}"
    )
    if result.rank_time:
        print(f"  排行：{result.rank_time}")
    if not result.hits:
        print("  目前沒有符合條件的標的。")
        return
    for hit in result.hits:
        s, snap = hit.stock, hit.snapshot
        chg = f"{s.change_percent:+.2f}%" if s.change_percent is not None else ""
        print(
            f"  #{s.rank:3d} {s.name:8s} {s.symbol:10s} "
            f"{s.price:8.2f} {chg:>8s}  "
            f"收 {snap.close:.2f} < MA200 {snap.ma200:.2f}  "
            f"MA {snap.ma5:.2f}/{snap.ma10:.2f}/{snap.ma20:.2f}  "
            f"{snap.timestamp.strftime('%H:%M')}"
        )


def notify_new(result, seen: set[str]) -> None:
    fresh = [h for h in result.hits if hit_key(h) not in seen]
    for hit in result.hits:
        seen.add(hit_key(hit))
    if not fresh:
        return
    title, body = format_hit_message([(h.stock, h.snapshot) for h in fresh])
    sent = send_notifications(title, body)
    print(f"  通知 {len(fresh)} 則 → {', '.join(sent) or '未設定通道（填 TELEGRAM）'}")


def main() -> int:
    args = parse_args()
    cfg = _config(args)
    seen: set[str] = set()

    if args.test:
        result = run_scan(cfg)
        print_result(result)
        title, body = format_hit_message([(h.stock, h.snapshot) for h in result.hits])
        sent = send_notifications(title, body)
        print(f"測試通知 → {', '.join(sent) or '未設定通道（填 TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID）'}")
        return 0

    print("五分K空頭監視：MA5 < MA10 < MA20、跌破 MA200、15分／小時K < MA20。Ctrl+C 結束。", flush=True)
    while True:
        if not market_open():
            print("盤外等待…", flush=True)
            time.sleep(max(30, args.interval))
            continue
        try:
            result = run_scan(cfg)
            print_result(result)
            notify_new(result, seen)
        except Exception as exc:  # noqa: BLE001
            print(f"掃描失敗：{exc}", flush=True)
        time.sleep(max(20, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())

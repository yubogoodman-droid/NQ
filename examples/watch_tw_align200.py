#!/usr/bin/env python3
"""一分K：5>10>20>60 且站上 MA200 → Telegram 通知。

掃成交額前 100。在下面填 Telegram 後執行：

    python3 examples/watch_tw_align200.py

Ctrl+C 結束。同一檔同一根 K 不重發。
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import date
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nq.align200 import detect_align200, format_alert
from nq.yahoo_1m import fetch_1m
from tw.ranking import fetch_turnover_ranking, filter_etfs

TELEGRAM_BOT_TOKEN = ""
TELEGRAM_CHAT_ID = ""
SEEN_PATH = Path(__file__).resolve().parents[1] / "output" / "align200_seen.json"


def apply_keys() -> None:
    if TELEGRAM_BOT_TOKEN.strip():
        os.environ.setdefault("TELEGRAM_BOT_TOKEN", TELEGRAM_BOT_TOKEN.strip())
    if TELEGRAM_CHAT_ID.strip():
        os.environ.setdefault("TELEGRAM_CHAT_ID", TELEGRAM_CHAT_ID.strip())


def load_seen() -> set[str]:
    if not SEEN_PATH.exists():
        return set()
    try:
        return set(json.loads(SEEN_PATH.read_text()))
    except Exception:
        return set()


def save_seen(seen: set[str]) -> None:
    SEEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    SEEN_PATH.write_text(json.dumps(sorted(seen)))


def telegram(text: str) -> bool:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat:
        print(text, flush=True)
        return False
    r = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data={"chat_id": chat, "text": text},
        timeout=20,
    )
    return r.ok


def scan_once(top: int = 100) -> int:
    stocks, label = fetch_turnover_ranking(top=max(top * 2, 150))
    stocks = filter_etfs(stocks)[:top]
    print(f"{label} 掃描 {len(stocks)} 檔", flush=True)
    seen = load_seen()
    sent = 0
    today = date.today()
    for stock in stocks:
        try:
            df = fetch_1m(stock.symbol, period="5d")
        except Exception as exc:
            print(f"略過 {stock.symbol}: {exc}", flush=True)
            continue
        sigs = detect_align200(df, symbol=stock.symbol, name=stock.name, on_date=today)
        for sig in sigs:
            key = f"{sig.symbol}:{sig.timestamp}"
            if key in seen:
                continue
            if telegram(format_alert(sig)):
                print(f"已推 {key}", flush=True)
            else:
                print(f"通知（未設 Telegram）\n{format_alert(sig)}", flush=True)
            seen.add(key)
            sent += 1
    save_seen(seen)
    return sent


def main() -> None:
    apply_keys()
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--once", action="store_true")
    p.add_argument("--interval", type=int, default=60)
    p.add_argument("--top", type=int, default=100)
    p.add_argument("--test", action="store_true")
    args = p.parse_args()
    if args.test:
        ok = telegram("align200 監看測試：如果你看到這則，Telegram 已通。")
        print("Telegram", "OK" if ok else "未設定或失敗")
        return
    if args.once:
        print(f"送出 {scan_once(args.top)} 則")
        return
    print("watch 中，每根 1m 收盤掃一次（Ctrl+C 停）", flush=True)
    while True:
        try:
            scan_once(args.top)
        except Exception as exc:
            print(f"掃描失敗：{exc}", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""PyCharm 監看腳本：符合 ORCL 形 1m 空就跳通知。

用法：
  1. 用 PyCharm 打開這個檔
  2. 改下面「改這裡」
  3. 右上角綠三角 Run（或右鍵 → Run 'pycharm_orcl_watch'）
  4. 第一次先把 TEST_ONLY 設 True，確認電腦彈窗 / Telegram 有來
  5. 再改回 False，讓它每根 1 分鐘收盤掃一次

停止：PyCharm 紅方塊 Stop，或終端機 Ctrl+C。

通知：
  - 電腦右下角彈窗（Windows / macOS / Linux）
  - Telegram（填 token，或專案根目錄 tg_config.env）
"""
from __future__ import annotations

import sys
from pathlib import Path

# ========================= 改這裡 =========================
WATCH_ALL = True                 # True = 掃幣安所有 USDT 永續（含 ORCL）
SYMBOLS = "ORCLUSDT"             # WATCH_ALL=False 時只看這些，逗號分隔
TELEGRAM_BOT_TOKEN = ""          # BotFather，例如 123456:ABC...
TELEGRAM_CHAT_ID = ""            # 你的 chat id，數字
TEST_ONLY = False                # True = 只測通知通不通，然後結束
DRY_RUN = False                  # True = 符合也只印，不送 Telegram（彈窗仍會跳）
ONCE = False                     # True = 只掃當下這一分，然後結束
DESKTOP_POPUP = True             # 電腦彈窗
WORKERS = 12                     # 並行下載 K 線
# ========================================================

sys.path.insert(0, str(Path(__file__).resolve().parent))

from watch_orcl_death_cross import run_pycharm  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(
        run_pycharm(
            watch_all=WATCH_ALL,
            symbols=SYMBOLS,
            dry_run=DRY_RUN,
            test_only=TEST_ONLY,
            once=ONCE,
            desktop=DESKTOP_POPUP,
            workers=WORKERS,
            telegram_bot_token=TELEGRAM_BOT_TOKEN,
            telegram_chat_id=TELEGRAM_CHAT_ID,
        )
    )

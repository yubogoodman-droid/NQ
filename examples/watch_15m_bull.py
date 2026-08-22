#!/usr/bin/env python3
"""依回測同一套規則推 Telegram。入口在 15M多排.py / nq.watch_ma_bull。

    python 15M多排.py --test
    python 15M多排.py
    python examples/watch_15m_bull.py --tf 1h
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

TELEGRAM_BOT_TOKEN = ""
TELEGRAM_CHAT_ID = ""


def add_nq_to_path() -> Path:
    here = Path(__file__).resolve().parent
    for folder in (here, *here.parents):
        if (folder / "nq" / "binance.py").exists():
            root = str(folder)
            if root not in sys.path:
                sys.path.insert(0, root)
            return folder
    raise SystemExit("找不到 nq。請在專案根目錄執行，或把 nq 資料夾放在腳本同一層。")


if TELEGRAM_BOT_TOKEN.strip():
    os.environ["TELEGRAM_BOT_TOKEN"] = TELEGRAM_BOT_TOKEN.strip()
if TELEGRAM_CHAT_ID.strip():
    os.environ["TELEGRAM_CHAT_ID"] = TELEGRAM_CHAT_ID.strip()

add_nq_to_path()

from nq.watch_ma_bull import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())

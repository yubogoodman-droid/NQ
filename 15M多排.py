#!/usr/bin/env python3
"""15 分 / 1 小時 MA200 剛站上 → Telegram。

15m：剛站上，或站上後 4 根內才收出 7>25。
1h：只推本根剛站上 1h MA200（同樣 7>25）。

PyCharm 請把「整個 NQ 專案」打開，或把本檔和 nq 資料夾放同一層：

    PythonProject2/
      15M多排.py
      nq/
        __init__.py
        binance.py
        ma15_bull.py
        watch_ma_bull.py

再 pip install numpy requests matplotlib
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# —— 填這裡 ——
TELEGRAM_BOT_TOKEN = ""  # BotFather 給的 token，例如 123456:ABC...
TELEGRAM_CHAT_ID = ""    # 你的 chat id，數字


def add_nq_to_path() -> Path:
    here = Path(__file__).resolve().parent
    for folder in (here, *here.parents):
        if (folder / "nq" / "binance.py").exists():
            root = str(folder)
            if root not in sys.path:
                sys.path.insert(0, root)
            return folder
    raise SystemExit(
        "找不到 nq 套件（No module named 'nq'）。\n"
        "不要只拷這一個 py。請把 nq 資料夾和本檔放同一層，例如：\n"
        "  C:\\Users\\yubogood\\PycharmProjects\\PythonProject2\\15M多排.py\n"
        "  C:\\Users\\yubogood\\PycharmProjects\\PythonProject2\\nq\\binance.py\n"
        "或用 PyCharm 打開整個 NQ 專案再跑 15M多排.py。"
    )


if TELEGRAM_BOT_TOKEN.strip():
    os.environ["TELEGRAM_BOT_TOKEN"] = TELEGRAM_BOT_TOKEN.strip()
if TELEGRAM_CHAT_ID.strip():
    os.environ["TELEGRAM_CHAT_ID"] = TELEGRAM_CHAT_ID.strip()

add_nq_to_path()

from nq.watch_ma_bull import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())

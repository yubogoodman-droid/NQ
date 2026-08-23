#!/usr/bin/env python3
"""同一個腳本同時監看 15 分 K 與 1 小時 K，符合條件就推 Telegram。

15m：本根剛站上 15m MA200（收盤 > MA7 > MA25），之後連續 3 根收盤都在 MA200 上，
     且當時 1h MA25 未下彎、1h 收盤 > MA7 > MA25。第 3 根收完才推。
1h ：本根剛站上 1h MA200（收盤 > MA7 > MA25），且 1h MA25 未下彎。本根收完就推。
     不要求連 3 根。4h 圖只對照、不擋單。

用法（PyCharm 按 Run 即可，視窗不要關）：

    pip install numpy requests matplotlib
    python 15M多排.py --test          # 先測 Telegram 通不通
    python 15M多排.py                 # 15m + 1h 一直監看
    python 15M多排.py --once          # 只掃一輪
    python 15M多排.py --tf 15m        # 只看 15 分
    python 15M多排.py --tf 1h         # 只看 1 小時

nq 資料夾必須和本檔同一層（或打開整個 NQ 專案）：

    PythonProject2/
      15M多排.py
      nq/
      telegram_local.py   # 可選，和下面填 token 二選一
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# —— 沒有 telegram_local.py 就填這裡 ——
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


def load_telegram_keys(root: Path) -> None:
    if TELEGRAM_BOT_TOKEN.strip():
        os.environ["TELEGRAM_BOT_TOKEN"] = TELEGRAM_BOT_TOKEN.strip()
    if TELEGRAM_CHAT_ID.strip():
        os.environ["TELEGRAM_CHAT_ID"] = TELEGRAM_CHAT_ID.strip()
    if os.environ.get("TELEGRAM_BOT_TOKEN") and os.environ.get("TELEGRAM_CHAT_ID"):
        return
    for folder in (Path(__file__).resolve().parent, root, Path.cwd()):
        f = folder / "telegram_local.py"
        if not f.is_file():
            continue
        ns: dict = {}
        exec(f.read_text(encoding="utf-8"), ns)
        tok = str(ns.get("TELEGRAM_BOT_TOKEN", "")).strip()
        chat = str(ns.get("TELEGRAM_CHAT_ID", "")).strip()
        if tok:
            os.environ.setdefault("TELEGRAM_BOT_TOKEN", tok)
        if chat:
            os.environ.setdefault("TELEGRAM_CHAT_ID", chat)
        return


root = add_nq_to_path()
load_telegram_keys(root)

from nq.watch_ma_bull import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())

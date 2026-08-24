#!/usr/bin/env python3
"""幣安 1 分 K：7>14>25>99 多頭排列上站 MA200 → Telegram。

用法（PyCharm 按 Run 即可，視窗不要關）：

    pip install numpy requests matplotlib
    python 一分K多排.py --test          # 先測 Telegram 通不通
    python 一分K多排.py                 # 一直監看成交額前 50
    python 一分K多排.py --once          # 只掃一輪
    python 一分K多排.py backtest --today --pages

在下面填 token，或旁邊放 telegram_local.py / tg_config.env。
nq 資料夾必須和本檔同一層。
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
        if (folder / "nq" / "ma1m_bull.py").exists():
            root = str(folder)
            if root not in sys.path:
                sys.path.insert(0, root)
            return folder
    raise SystemExit(
        "找不到 nq 套件。請把 nq 資料夾和本檔放同一層，或打開整個 NQ 專案再跑。"
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

sys.path.insert(0, str(root / "examples"))
from binance_1m_bull import main  # noqa: E402

if __name__ == "__main__":
    argv = sys.argv[1:]
    if not argv:
        argv = ["alert"]
    elif argv[0] not in {"alert", "backtest"}:
        argv = ["alert", *argv]
    raise SystemExit(main(argv))

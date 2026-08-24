#!/usr/bin/env python3
"""相容入口：實際程式在專案根目錄 watch_tw.py（單一檔，不依賴 tw 套件）。"""

from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parents[1] / "watch_tw.py"), run_name="__main__")

#!/usr/bin/env python3
"""PyCharm 請執行這個檔（專案根目錄）。等同 examples/watch_tw_shioaji.py。"""

from pathlib import Path
import runpy

runpy.run_path(
    str(Path(__file__).resolve().parent / "examples" / "watch_tw_shioaji.py"),
    run_name="__main__",
)

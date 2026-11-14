#!/usr/bin/env python3
"""15m 200MA 附近進場（ETH 截圖線型）。參數與 scan_binance_15m_align.py 相同。"""
from __future__ import annotations

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).with_name("scan_binance_15m_align.py")), run_name="__main__")

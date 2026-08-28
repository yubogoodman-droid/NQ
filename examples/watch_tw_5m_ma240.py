#!/usr/bin/env python3
"""已改成一小時 K 收盤站上 MA60。此檔轉去 watch_tw_1h_ma60.py。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from watch_tw_1h_ma60 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())

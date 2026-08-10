#!/usr/bin/env python3
"""Deprecated wrapper: use backtest_tiers_1d.py. Kept for compatibility."""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("backtest_tiers_1d.py")), run_name="__main__")

#!/usr/bin/env python3
"""NQ 五分 W 底：對齊 9/15 截圖那種，淺雙底與隔很久才破的不算。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nq.patterns import detect_w_bottoms  # noqa: E402
from nq.strategy import NQWBottomStrategy  # noqa: E402
from run_backtest import make_sample_w_bottom_bars  # noqa: E402


def _ohlc(close: np.ndarray, low: np.ndarray | None = None) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    n = len(close)
    if low is None:
        low = close - 2.0
    high = np.maximum(close, np.roll(close, 1)) + 2.0
    high[0] = close[0] + 2.0
    idx = pd.date_range("2026-09-15 00:00", periods=n, freq="5min", tz="America/New_York")
    return pd.DataFrame(
        {"open": np.r_[close[0], close[:-1]], "high": high, "low": low, "close": close, "volume": 1},
        index=idx,
    )


def make_screenshot_like() -> pd.DataFrame:
    """先殺 0.8%，兩個相近谷，4 根內破頸線。"""
    n = 80
    close = np.full(n, 29450.0)
    low = close - 4.0
    # dump into L1
    for i, px in enumerate(np.linspace(29450, 29240, 28)):
        close[i] = px
        low[i] = px - 6.0
    low[27] = 29231.0
    close[27] = 29240.0
    # bounce / neck
    for i, px in enumerate(np.linspace(29255, 29290, 6), start=28):
        close[i] = px
        low[i] = px - 4.0
    close[33] = 29288.0
    # right leg to L2
    for i, px in enumerate(np.linspace(29280, 29245, 8), start=34):
        close[i] = px
        low[i] = px - 4.0
    low[41] = 29236.5
    close[41] = 29250.0
    # breakout
    close[45] = 29300.0
    low[45] = 29280.0
    close[46:] = 29320.0
    high = np.maximum(close, np.roll(close, 1)) + 3.0
    high[0] = close[0] + 3.0
    high[33] = 29295.0
    idx = pd.date_range("2026-09-15 01:00", periods=n, freq="5min", tz="America/New_York")
    return pd.DataFrame(
        {"open": np.r_[close[0], close[:-1]], "high": high, "low": low, "close": close, "volume": 1},
        index=idx,
    )


def make_tiny_chop() -> pd.DataFrame:
    """0.2% 淺雙底，不該進。"""
    n = 80
    close = np.full(n, 29450.0)
    low = close - 2.0
    close[20:28] = 29420.0
    low[24] = 29410.0
    close[28:34] = 29435.0
    close[34:42] = 29422.0
    low[38] = 29412.0
    close[45:] = 29450.0
    return _ohlc(close, low)


def make_morning_w() -> pd.DataFrame:
    """同一形狀但發生在 07:xx 早盤，不該進。"""
    df = make_screenshot_like()
    df.index = pd.date_range("2026-09-09 05:00", periods=len(df), freq="5min", tz="America/New_York")
    return df


def test_screenshot_like_fires() -> None:
    df = make_screenshot_like()
    sigs = NQWBottomStrategy().generate_signals(df)
    assert sigs, "截圖那種 W 底應該進場"
    p = sigs[0].pattern
    assert abs(p.first_low - 29231.0) < 8
    assert abs(p.second_low - 29236.5) < 12


def test_morning_w_rejected() -> None:
    assert NQWBottomStrategy().generate_signals(make_morning_w()) == []


def test_tiny_chop_rejected() -> None:
    assert NQWBottomStrategy().generate_signals(make_tiny_chop()) == []


def test_loose_still_sees_demo_w() -> None:
    df = make_sample_w_bottom_bars()
    assert detect_w_bottoms(df)
    assert NQWBottomStrategy.loose().generate_signals(df)


def test_strict_skips_demo_chop_scale() -> None:
    df = make_sample_w_bottom_bars()
    assert NQWBottomStrategy().generate_signals(df) == []


if __name__ == "__main__":
    test_screenshot_like_fires()
    test_morning_w_rejected()
    test_tiny_chop_rejected()
    test_loose_still_sees_demo_w()
    test_strict_skips_demo_chop_scale()
    print("ok")

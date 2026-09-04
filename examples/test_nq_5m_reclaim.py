#!/usr/bin/env python3
"""Synthetic tests for NQ 5m 破底後 30 分內 5/10/20 多排。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nq_5m_base_stack import ET, simulate, sma  # noqa: E402
from nq_5m_reclaim import detect_signals  # noqa: E402


def _bars(*, lift_delay: int = 1, n: int = 90) -> pd.DataFrame:
    """Flat 2h, one deep spike through the floor, then a fast lift that fans 5/10/20."""
    close = np.full(n, 20000.0)
    for i in range(1, 50):
        close[i] = 20000.0 + (0.3 if i % 2 == 0 else -0.2)
    close[50] = 19955.0
    for k, i in enumerate(range(51, n)):
        if k < lift_delay:
            close[i] = 19960.0
        else:
            close[i] = 20020.0 + 12.0 * (k - lift_delay)
    high = close + 3.0
    low = close - 3.0
    low[50] = 19948.0
    open_ = np.r_[close[0], close[:-1]]
    idx = pd.date_range("2026-09-01 09:00", periods=n, freq="5min", tz=ET)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": np.full(n, 80.0)},
        index=idx,
    )


def test_reclaim_stack_within_30m() -> None:
    df = _bars(lift_delay=1)
    sigs = detect_signals(df)
    assert sigs, "should take a 5/10/20 stack reclaim inside 30 minutes"
    assert sigs[0].entry_idx - sigs[0].dump_idx <= 6
    assert sigs[0].entry_price > sigs[0].ma5
    assert sigs[0].ma5 > sigs[0].ma10 > sigs[0].ma20
    assert simulate(df, sigs)


def test_late_reclaim_rejected() -> None:
    df = _bars(lift_delay=10)
    assert not detect_signals(df), "reclaim after 30 minutes should not fill"


def test_close_above_mas_without_stack_rejected() -> None:
    """Price pops above all three MAs but they are still 20>10>5."""
    n = 80
    close = np.full(n, 20000.0)
    close[:40] = np.linspace(20200.0, 20000.0, 40)
    close[40] = 19920.0
    close[41:] = 20010.0
    high = close + 2.0
    low = close - 2.0
    low[40] = 19905.0
    open_ = np.r_[close[0], close[:-1]]
    idx = pd.date_range("2026-09-01 09:00", periods=n, freq="5min", tz=ET)
    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": np.full(n, 80.0)},
        index=idx,
    )
    c = df["Close"].to_numpy(float)
    ma5, ma10, ma20 = sma(c, 5), sma(c, 10), sma(c, 20)
    # After the pop, MAs should still be inverted on the first reclaim bars.
    assert not detect_signals(df), "stand-back without 5>10>20 should not fill"


def test_no_break_no_trade() -> None:
    close = np.linspace(20000.0, 20080.0, 80)
    idx = pd.date_range("2026-09-01 09:00", periods=80, freq="5min", tz=ET)
    df = pd.DataFrame(
        {
            "Open": close - 1,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Volume": np.full(80, 80.0),
        },
        index=idx,
    )
    assert not detect_signals(df)


def main() -> int:
    test_reclaim_stack_within_30m()
    test_late_reclaim_rejected()
    test_close_above_mas_without_stack_rejected()
    test_no_break_no_trade()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

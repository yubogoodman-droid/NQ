#!/usr/bin/env python3
"""Synthetic tests for NQ 5m 破底後 30 分內站回 5/10/20."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nq_5m_base_stack import ET, simulate  # noqa: E402
from nq_5m_reclaim import detect_signals  # noqa: E402


def _bars(*, reclaim_at: int = 2, n: int = 80) -> pd.DataFrame:
    """Flat, then break a 2h floor, then lift above the MAs at reclaim_at bars after the low."""
    close = np.full(n, 20000.0)
    for i in range(1, 40):
        close[i] = 20000.0 + (0.4 if i % 2 == 0 else -0.2)
    floor = 19940.0
    close[40:45] = np.linspace(19980.0, floor, 5)
    close[45] = floor - 20.0
    for k, i in enumerate(range(46, n)):
        if k + 1 < reclaim_at:
            close[i] = floor - 8.0 + (k % 2)
        else:
            close[i] = 20020.0 + 2.0 * (k + 1 - reclaim_at)
    high = close + 2.0
    low = close - 2.0
    low[45] = floor - 22.0
    open_ = np.r_[close[0], close[:-1]]
    idx = pd.date_range("2026-09-01 09:00", periods=n, freq="5min", tz=ET)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": np.full(n, 80.0)},
        index=idx,
    )


def test_reclaim_within_30m() -> None:
    df = _bars(reclaim_at=2)
    sigs = detect_signals(df)
    assert sigs, "should take a reclaim inside 30 minutes"
    assert sigs[0].entry_idx - sigs[0].dump_idx <= 6
    assert sigs[0].entry_price > sigs[0].ma5
    assert sigs[0].entry_price > sigs[0].ma20
    trades = simulate(df, sigs)
    assert trades


def test_late_reclaim_rejected() -> None:
    df = _bars(reclaim_at=10)
    assert not detect_signals(df), "reclaim after 30 minutes should not fill"


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
    test_reclaim_within_30m()
    test_late_reclaim_rejected()
    test_no_break_no_trade()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

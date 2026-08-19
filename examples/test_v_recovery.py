"""Synthetic path that forms a full bullish MA stack."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nq.ma_stack import add_indicators, count_stack_events, ladder_counts


def make_stack_bars(n: int = 400) -> pd.DataFrame:
    close = np.linspace(29400.0, 29720.0, n)
    noise = np.sin(np.linspace(0, 8, n)) * 1.5
    close = close + noise
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) + 1.2
    low = np.minimum(open_, close) - 1.2
    idx = pd.date_range("2026-08-18 16:00", periods=n, freq="1min", tz="America/New_York")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": np.full(n, 150.0)},
        index=idx,
    )


def test_full_stack_fires_once_per_cooldown() -> None:
    df = add_indicators(make_stack_bars())
    counts = ladder_counts(df, cooldown=30)
    assert counts["full"] >= 1
    sigs = count_stack_events(df, level="full", cooldown=30)
    assert sigs[0].order[0] == 5
    assert sigs[0].order[-1] == 200
    assert df["stack_full"].iloc[sigs[0].idx]
    assert df["above_all"].iloc[sigs[0].idx]


if __name__ == "__main__":
    test_full_stack_fires_once_per_cooldown()
    print("self-test ok")

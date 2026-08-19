"""Synthetic 1m path that contains one screenshot-like V-reclaim."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nq.v_recovery import STRICT, add_indicators, count_ladder


def make_v_recovery_bars(n: int = 320) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    close = np.full(n, 29550.0)
    high = close.copy()
    low = close.copy()
    open_ = close.copy()
    vol = np.full(n, 120.0)

    for i in range(1, n):
        close[i] = close[i - 1] + rng.normal(0, 1.2)
        open_[i] = close[i - 1]
        high[i] = max(open_[i], close[i]) + 1.5
        low[i] = min(open_[i], close[i]) - 1.5

    dump = 220
    open_[dump] = 29540.0
    high[dump] = 29542.0
    low[dump] = 29450.0
    close[dump] = 29470.0
    vol[dump] = 2500
    for i in range(dump + 1, dump + 35):
        step = 3.2
        open_[i] = close[i - 1]
        close[i] = close[i - 1] + step
        high[i] = close[i] + 1.0
        low[i] = open_[i] - 0.5
        vol[i] = 180

    idx = pd.date_range("2026-08-18 16:00", periods=n, freq="1min", tz="America/New_York")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )


def test_detects_one_strict_signal() -> None:
    df = add_indicators(make_v_recovery_bars())
    ladder = count_ladder(df, STRICT)
    assert ladder.dumps >= 1
    assert ladder.reclaim_fan == 1
    sig = ladder.signals[0]
    assert sig.dump.timestamp.strftime("%H:%M") == "19:40"
    assert sig.entry > sig.stop_loss


if __name__ == "__main__":
    test_detects_one_strict_signal()
    print("self-test ok")

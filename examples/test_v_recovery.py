"""Synthetic dump then bullish MA stack."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nq.ma_stack import STRICT_DUMP, add_indicators, dump_align_ladder, ladder_counts


def make_dump_then_stack(n: int = 420) -> pd.DataFrame:
    close = np.linspace(29520.0, 29640.0, n)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) + 1.2
    low = np.minimum(open_, close) - 1.2
    vol = np.full(n, 120.0)
    dump = 230
    open_[dump] = close[dump - 1]
    high[dump] = open_[dump] + 1
    low[dump] = open_[dump] - 85
    close[dump] = open_[dump] - 55
    vol[dump] = 2800
    for i in range(dump + 1, min(n, dump + 90)):
        open_[i] = close[i - 1]
        close[i] = close[i - 1] + 1.7
        high[i] = max(open_[i], close[i]) + 0.8
        low[i] = min(open_[i], close[i]) - 0.4
        vol[i] = 180
    idx = pd.date_range("2026-08-18 16:00", periods=n, freq="1min", tz="America/New_York")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )


def test_dump_plus_stack() -> None:
    df = add_indicators(make_dump_then_stack())
    ladder = dump_align_ladder(df, STRICT_DUMP, max_bars=120)
    assert ladder["dumps"] >= 1
    assert ladder["short"] >= 1
    sig = ladder["signals"][0]
    assert sig.short is not None
    assert sig.short.idx > sig.dump.idx


def test_align_only_still_counts() -> None:
    close = np.linspace(29400.0, 29720.0, 400)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    idx = pd.date_range("2026-08-18 16:00", periods=400, freq="1min", tz="America/New_York")
    df = add_indicators(
        pd.DataFrame(
            {
                "open": open_,
                "high": np.maximum(open_, close) + 1.2,
                "low": np.minimum(open_, close) - 1.2,
                "close": close,
                "volume": np.full(400, 150.0),
            },
            index=idx,
        )
    )
    assert ladder_counts(df)["full"] >= 1


if __name__ == "__main__":
    test_dump_plus_stack()
    test_align_only_still_counts()
    print("self-test ok")

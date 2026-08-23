#!/usr/bin/env python3
"""TW MA Reclaim scanner helpers (no network)."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scan_tw_ma_reclaim import (  # noqa: E402
    TPE,
    last_tw_session_yyyymmdd,
    session_mask,
    tw_pt_scale,
    yahoo_symbol,
)


def test_yahoo_symbol() -> None:
    assert yahoo_symbol("2330", "tse") == "2330.TW"
    assert yahoo_symbol("6488", "otc") == "6488.TWO"


def test_tw_pt_scale() -> None:
    assert abs(tw_pt_scale(20000) - 1.0) < 1e-9
    assert abs(tw_pt_scale(1000) - 0.05) < 1e-9
    assert tw_pt_scale(1.0) >= 1e-4


def test_session_mask() -> None:
    idx = pd.DatetimeIndex(
        [
            "2026-08-21 08:59",
            "2026-08-21 09:00",
            "2026-08-21 13:30",
            "2026-08-21 13:31",
        ],
        tz=TPE,
    )
    m = session_mask(idx)
    assert list(m) == [False, True, True, False]


def test_last_session_skips_weekend() -> None:
    sunday = datetime(2026, 8, 23, 12, 0, tzinfo=TPE)
    assert last_tw_session_yyyymmdd(sunday) == "20260821"


def main() -> int:
    test_yahoo_symbol()
    test_tw_pt_scale()
    test_session_mask()
    test_last_session_skips_weekend()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

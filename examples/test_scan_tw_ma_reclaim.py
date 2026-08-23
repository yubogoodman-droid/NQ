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
    _is_stock_code,
    filter_by_max_price,
    last_tw_session_yyyymmdd,
    session_mask,
    tw_pt_scale,
    yahoo_symbol,
)


def test_is_stock_code() -> None:
    assert _is_stock_code("2330")
    assert _is_stock_code("8358")
    assert not _is_stock_code("0050")
    assert not _is_stock_code("00937B")
    assert not _is_stock_code("00400A")


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


def test_filter_by_max_price() -> None:
    rows = [
        {"code": "2330", "close": 1400.0},
        {"code": "2408", "close": 500.0},
        {"code": "2303", "close": 55.0},
        {"code": "3008", "close": 2500.0},
    ]
    kept, dropped = filter_by_max_price(rows, 600.0, 100)
    assert [r["code"] for r in kept] == ["2408", "2303"]
    assert {r["code"] for r in dropped} == {"2330", "3008"}
    assert kept[0]["rank"] == 1


def test_last_session_skips_weekend() -> None:
    sunday = datetime(2026, 8, 23, 12, 0, tzinfo=TPE)
    assert last_tw_session_yyyymmdd(sunday) == "20260821"
    monday_morning = datetime(2026, 8, 24, 10, 0, tzinfo=TPE)
    assert last_tw_session_yyyymmdd(monday_morning) == "20260821"


def main() -> int:
    test_is_stock_code()
    test_yahoo_symbol()
    test_tw_pt_scale()
    test_session_mask()
    test_filter_by_max_price()
    test_last_session_skips_weekend()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""離線測試：15m 同時跌破 MA7/14/25/200、美東偵測窗。"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from watch_binance_equity_ma_break import (  # noqa: E402
    ET,
    MIN_DEPTH_PCT,
    bar_in_session,
    below_all,
    is_fresh_break,
    is_signal,
    next_window_start,
    now_should_scan,
    short_fwd_pct,
    sma,
)


def test_sma() -> None:
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = sma(arr, 3)
    assert np.isnan(out[1])
    assert abs(out[2] - 2.0) < 1e-9
    assert abs(out[4] - 4.0) < 1e-9


def _bars(close: np.ndarray, *, open_: np.ndarray | None = None) -> dict:
    n = len(close)
    close = close.astype(float)
    o = close.copy() if open_ is None else open_.astype(float)
    d = {
        "t": np.arange(n, dtype=np.int64) * 900_000,
        "c": close,
        "o": o,
        "h": np.maximum(o, close),
        "l": np.minimum(o, close),
        "v": np.ones(n),
    }
    for nma in (7, 14, 25, 200):
        d[f"m{nma}"] = sma(d["c"], nma)
    return d


def test_fresh_break_first_close_below_all() -> None:
    close = np.full(220, 100.0)
    # 讓均線在 100 附近，最後一根砸到 90
    close[-1] = 90.0
    d = _bars(close)
    assert not below_all(d, 218)
    assert below_all(d, 219)
    assert is_fresh_break(d, 219)
    assert not is_fresh_break(d, 218)


def test_already_below_is_not_fresh() -> None:
    close = np.full(220, 100.0)
    close[-3:] = 90.0
    d = _bars(close)
    assert below_all(d, 218)
    assert below_all(d, 219)
    assert not is_fresh_break(d, 219)


def test_kiss_not_signal_smash_is() -> None:
    close = np.full(220, 100.0)
    o = close.copy()
    close[-1] = 99.8  # ~0.2%，不夠深
    o[-1] = 100.2
    d = _bars(close, open_=o)
    assert is_fresh_break(d, 219)
    assert not is_signal(d, 219)

    close = np.full(220, 100.0)
    o = close.copy()
    close[-1] = 99.0  # 1%
    o[-1] = 100.5
    d = _bars(close, open_=o)
    assert is_signal(d, 219)
    assert MIN_DEPTH_PCT == 0.40


def test_short_fwd_pct() -> None:
    assert abs(short_fwd_pct(100.0, 97.0) - 3.0) < 1e-9
    assert abs(short_fwd_pct(100.0, 104.0) + 4.0) < 1e-9


def test_session_window_weekday() -> None:
    # 2026-09-09 星期三。15m 開盤 ms → 收盤美東時間
    et = ZoneInfo("America/New_York")

    def open_ms(h: int, m: int) -> int:
        close = datetime(2026, 9, 9, h, m, tzinfo=et)
        open_ = close.timestamp() - 900
        return int(open_ * 1000)

    assert bar_in_session(open_ms(9, 0))  # 08:45–09:00 收盤，開始偵測
    assert bar_in_session(open_ms(9, 30))
    assert bar_in_session(open_ms(16, 0))
    assert not bar_in_session(open_ms(8, 45))
    assert not bar_in_session(open_ms(16, 15))


def test_weekend_off() -> None:
    et = ZoneInfo("America/New_York")
    sat = datetime(2026, 9, 12, 10, 0, tzinfo=et)
    open_ms = int((sat.timestamp() - 900) * 1000)
    assert not bar_in_session(open_ms)
    assert not now_should_scan(sat)


def test_now_should_scan_and_next_window() -> None:
    et = ET
    before = datetime(2026, 9, 9, 8, 59, tzinfo=et)
    assert not now_should_scan(before)
    nxt = next_window_start(before)
    assert nxt.hour == 9 and nxt.minute == 0 and nxt.day == 9

    inside = datetime(2026, 9, 9, 12, 0, tzinfo=et)
    assert now_should_scan(inside)

    after = datetime(2026, 9, 9, 16, 1, tzinfo=et)
    assert not now_should_scan(after)
    nxt = next_window_start(after)
    assert nxt.day == 10 and nxt.hour == 9

    friday_after = datetime(2026, 9, 11, 16, 1, tzinfo=et)
    nxt = next_window_start(friday_after)
    assert nxt.weekday() == 0 and nxt.hour == 9  # 下週一


def main() -> int:
    test_sma()
    test_fresh_break_first_close_below_all()
    test_already_below_is_not_fresh()
    test_kiss_not_signal_smash_is()
    test_short_fwd_pct()
    test_session_window_weekday()
    test_weekend_off()
    test_now_should_scan_and_next_window()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

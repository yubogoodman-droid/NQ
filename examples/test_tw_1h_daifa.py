#!/usr/bin/env python3
"""Synthetic tests for 台股 1h 達發多（不打 Yahoo）。"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tw_1h_daifa import (  # noqa: E402
    detect_signals,
    filter_entry_window,
    simulate,
    summarize_trades,
)
from tw_1h_reclaim import TPE  # noqa: E402

HOURS = (9, 10, 11, 12, 13)


def session_index(n: int, start: str = "2026-06-01 09:00") -> pd.DatetimeIndex:
    t0 = pd.Timestamp(start, tz=TPE)
    d = t0.date()
    hi = HOURS.index(t0.hour) if t0.hour in HOURS else 0
    times = []
    while len(times) < n:
        if d.weekday() < 5:
            times.append(datetime(d.year, d.month, d.day, HOURS[hi], 0, tzinfo=TPE))
            hi += 1
            if hi >= len(HOURS):
                hi = 0
                d += timedelta(days=1)
        else:
            d += timedelta(days=1)
            hi = 0
    return pd.DatetimeIndex(times)


def make_df(opens, highs, lows, closes) -> pd.DataFrame:
    n = len(closes)
    return pd.DataFrame(
        {
            "Open": np.asarray(opens, dtype=float),
            "High": np.asarray(highs, dtype=float),
            "Low": np.asarray(lows, dtype=float),
            "Close": np.asarray(closes, dtype=float),
            "Volume": np.full(n, 1000.0),
        },
        index=session_index(n),
    )


def flat(n: int = 80):
    closes = np.full(n, 100.0)
    opens = closes.copy()
    highs = closes + 0.4
    lows = closes - 0.4
    return opens, highs, lows, closes


def paint_daifa(opens, highs, lows, closes, i: int = 70) -> None:
    """前收在均線下，開盤棒跳空且整根在 MA60 上。"""
    opens[i - 1], highs[i - 1], lows[i - 1], closes[i - 1] = 99.0, 99.2, 96.0, 97.0
    opens[i], highs[i], lows[i], closes[i] = 103.0, 106.0, 104.0, 105.0


def test_opening_gap_stands_on_ma60() -> None:
    opens, highs, lows, closes = flat()
    paint_daifa(opens, highs, lows, closes, 70)
    assert session_index(80)[70].hour == 9
    df = make_df(opens, highs, lows, closes)
    funnel: dict = {}
    sigs = detect_signals(df, funnel=funnel)
    assert funnel.get("entry") == 1
    assert len(sigs) == 1
    s = sigs[0]
    assert s.entry_idx == 70
    assert s.entry_price == 105.0
    assert s.bar_low == 104.0
    assert s.prev_high == 99.2
    assert s.prev_close == 97.0
    assert s.bar_low > s.ma60
    assert s.prev_close < s.ma60
    assert s.ma5 > s.ma10 > s.ma20
    assert s.gap_pct > 0


def test_second_bar_of_session_ignored() -> None:
    opens, highs, lows, closes = flat()
    paint_daifa(opens, highs, lows, closes, 71)
    df = make_df(opens, highs, lows, closes)
    assert df.index[71].hour == 10
    funnel: dict = {}
    sigs = detect_signals(df, funnel=funnel)
    assert sigs == []
    assert funnel.get("entry", 0) == 0


def test_overlap_is_not_a_gap() -> None:
    opens, highs, lows, closes = flat()
    paint_daifa(opens, highs, lows, closes, 70)
    highs[69] = 104.5  # 前高碰到這根低點 104，缺口補掉
    lows[70] = 104.0
    df = make_df(opens, highs, lows, closes)
    funnel: dict = {}
    sigs = detect_signals(df, funnel=funnel)
    assert sigs == []
    assert funnel.get("no_gap", 0) >= 1


def test_wick_through_ma60_rejected() -> None:
    """收盤在 MA60 上，但低點刺破，不算整根站上（達發 9/17 那種）。"""
    opens, highs, lows, closes = flat()
    paint_daifa(opens, highs, lows, closes, 70)
    highs[69] = 98.0
    lows[70] = 99.0  # 跳空仍在，但低點低於 MA60
    df = make_df(opens, highs, lows, closes)
    funnel: dict = {}
    sigs = detect_signals(df, funnel=funnel)
    assert sigs == []
    assert funnel.get("not_above_ma60", 0) >= 1


def test_already_above_ma60_rejected() -> None:
    opens, highs, lows, closes = flat()
    opens[69], highs[69], lows[69], closes[69] = 109.0, 110.5, 108.0, 110.0
    opens[70], highs[70], lows[70], closes[70] = 113.0, 116.0, 112.0, 114.0
    df = make_df(opens, highs, lows, closes)
    funnel: dict = {}
    sigs = detect_signals(df, funnel=funnel)
    assert sigs == []
    assert funnel.get("prev_above", 0) >= 1


def test_stack_required() -> None:
    opens, highs, lows, closes = flat()
    for j in (66, 67, 68):
        opens[j] = highs[j] = lows[j] = closes[j] = 70.0
        highs[j] = 70.4
        lows[j] = 69.6
    opens[69], highs[69], lows[69], closes[69] = 71.0, 73.0, 70.0, 72.0
    # 收高到整根站上 MA60，但前幾根太弱，MA5 仍低於 MA10。
    opens[70], highs[70], lows[70], closes[70] = 102.0, 112.0, 100.0, 110.0
    df = make_df(opens, highs, lows, closes)
    funnel: dict = {}
    sigs = detect_signals(df, funnel=funnel)
    assert sigs == []
    assert funnel.get("no_stack", 0) >= 1


def test_simulate_stop_and_target() -> None:
    opens, highs, lows, closes = flat(82)
    paint_daifa(opens, highs, lows, closes, 70)
    # 下一根打到停損（低點 104）
    opens[71], highs[71], lows[71], closes[71] = 104.5, 105.0, 103.0, 103.5
    df = make_df(opens, highs, lows, closes)
    sigs = detect_signals(df)
    assert len(sigs) == 1
    trades = simulate(df, sigs)
    assert trades[0].exit_reason == "stop"
    assert trades[0].pnl_points < 0

    opens, highs, lows, closes = flat(82)
    paint_daifa(opens, highs, lows, closes, 70)
    # 風險 = 105-104 = 1，2R = 107
    opens[71], highs[71], lows[71], closes[71] = 105.2, 108.0, 104.5, 107.5
    df = make_df(opens, highs, lows, closes)
    trades = simulate(df, detect_signals(df))
    assert trades[0].exit_reason == "target"
    assert abs(trades[0].target_price - 107.0) < 1e-9
    assert trades[0].pnl_points > 0


def test_filter_and_summary() -> None:
    opens, highs, lows, closes = flat()
    paint_daifa(opens, highs, lows, closes, 70)
    df = make_df(opens, highs, lows, closes)
    sigs = detect_signals(df)
    assert len(filter_entry_window(df, sigs, 14)) == 1
    assert filter_entry_window(df, sigs, 0) == list(sigs)
    stats = summarize_trades([])
    assert stats["count"] == 0
    assert stats["win_rate"] == 0.0


def main() -> int:
    test_opening_gap_stands_on_ma60()
    test_second_bar_of_session_ignored()
    test_overlap_is_not_a_gap()
    test_wick_through_ma60_rejected()
    test_already_above_ma60_rejected()
    test_stack_required()
    test_simulate_stop_and_target()
    test_filter_and_summary()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

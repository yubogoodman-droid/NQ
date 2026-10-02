#!/usr/bin/env python3
"""Synthetic tests for 台股 1h 突破後回踩 MA60（不打 Yahoo）。"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tw_1h_ma60_retest import (  # noqa: E402
    detect_signals,
    filter_entry_window,
    loose_params,
    simulate,
    sma,
    summarize_trades,
)

TPE = ZoneInfo("Asia/Taipei")
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


def ohlc_from_close(closes, lows=None, highs=None) -> pd.DataFrame:
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    if lows is None:
        lows = closes - 0.25
    if highs is None:
        highs = np.maximum(closes, np.asarray(lows, dtype=float)) + 0.25
    opens = np.concatenate([[closes[0]], closes[:-1]])
    return pd.DataFrame(
        {
            "Open": opens,
            "High": np.asarray(highs, dtype=float),
            "Low": np.asarray(lows, dtype=float),
            "Close": closes,
            "Volume": np.full(n, 1000.0),
        },
        index=session_index(n),
    )


def rising_base(n: int = 80) -> list[float]:
    """緩升，讓 MA60 略微上彎，收盤略高於均線但還沒大噴。"""
    return list(np.linspace(96.0, 100.0, n))


def test_sma() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
    ma = sma(x, 3)
    assert np.isnan(ma[1])
    assert abs(ma[2] - 2.0) < 1e-9


def test_breakout_then_ma60_retest_enters() -> None:
    """先拉離 MA60，再回踩均線附近收紅 → 進場。"""
    base = rising_base(80)
    ext = [102.0, 105.0, 108.0, 107.0, 105.5]
    closes = base + ext
    df0 = ohlc_from_close(closes)
    ma = sma(df0["Close"].to_numpy(float), 60)
    ma_now = float(ma[-1])
    assert 108.0 / ma_now - 1.0 >= 0.03

    # 回踩：低點碰到 MA60，收盤站回
    pb_close = ma_now * 1.008
    pb_low = ma_now * 0.997
    pb_high = max(pb_close, ma_now * 1.012)
    closes = closes + [pb_close]
    lows = list(df0["Low"]) + [pb_low]
    highs = list(df0["High"]) + [pb_high]
    df = ohlc_from_close(closes, lows=lows, highs=highs)

    funnel: dict = {}
    sigs = detect_signals(df, loose_params(), funnel=funnel)
    assert len(sigs) == 1, funnel
    s = sigs[0]
    assert s.entry_idx == len(df) - 1
    assert s.ext_pct >= 0.03
    assert s.entry_idx > s.peak_idx
    assert abs(s.ma60 - float(sma(df["Close"].to_numpy(float), 60)[-1])) < 1e-6


def test_no_extension_no_signal() -> None:
    """黏著 MA60 波動 < 3%，不算突破後回踩。"""
    closes = [100.0] * 70 + [100.4, 100.8, 100.5, 100.2, 100.1]
    df = ohlc_from_close(closes)
    funnel: dict = {}
    sigs = detect_signals(df, loose_params(), funnel=funnel)
    assert sigs == []
    assert funnel.get("entry", 0) == 0


def test_breakdown_is_not_retest() -> None:
    """突破後直接跌破 MA60 收盤，那是轉空，不是回踩。"""
    base = rising_base(80)
    ext = [102.0, 105.0, 108.0]
    crash = [104.0, 98.0]  # 收在均線下方很多
    df = ohlc_from_close(base + ext + crash)
    funnel: dict = {}
    sigs = detect_signals(df, loose_params(), funnel=funnel)
    assert sigs == []
    assert funnel.get("breakdown", 0) >= 1


def test_reclaim_from_below_is_not_this_setup() -> None:
    """先跌破再站回 = 破底翻，這套不吃。"""
    below = list(np.linspace(100.0, 92.0, 80))
    reclaim = [96.0, 98.0, 101.0, 103.0]
    df = ohlc_from_close(below + reclaim)
    sigs = detect_signals(df, loose_params())
    assert sigs == []


def test_chase_extended_close_rejected() -> None:
    """回踩那根如果收在遠離 MA60 的位置，是追價不是進場點。"""
    base = rising_base(80)
    ext = [102.0, 105.0, 108.0, 107.0]
    df0 = ohlc_from_close(base + ext)
    ma = sma(df0["Close"].to_numpy(float), 60)
    ma_now = float(ma[-1])
    # 長下影碰到 MA60，但收盤又回到 6% 外
    closes = base + ext + [ma_now * 1.06]
    lows = list(df0["Low"]) + [ma_now]
    highs = list(df0["High"]) + [ma_now * 1.07]
    df = ohlc_from_close(closes, lows=lows, highs=highs)
    sigs = detect_signals(df, loose_params())
    assert sigs == []


def test_first_retest_only() -> None:
    """同一段突破只吃第一次回踩。"""
    base = rising_base(80)
    ext = [102.0, 105.0, 108.0, 107.0]
    df0 = ohlc_from_close(base + ext)
    ma = sma(df0["Close"].to_numpy(float), 60)
    m = float(ma[-1])
    pb1 = m * 1.006
    pb2 = m * 1.005
    closes = base + ext + [pb1, m * 1.02, pb2]
    lows = list(df0["Low"]) + [m * 0.998, m * 1.01, m * 0.997]
    highs = list(df0["High"]) + [m * 1.015, m * 1.03, m * 1.012]
    df = ohlc_from_close(closes, lows=lows, highs=highs)
    sigs = detect_signals(df, loose_params())
    assert len(sigs) == 1
    assert sigs[0].entry_idx == len(base) + len(ext)


def test_same_bar_breakout_wick_not_entry() -> None:
    """突破那根本身的長下影不算回踩。"""
    base = rising_base(80)
    df0 = ohlc_from_close(base)
    ma = sma(df0["Close"].to_numpy(float), 60)
    m = float(ma[-1])
    closes = base + [m * 1.08]
    lows = list(df0["Low"]) + [m]
    highs = list(df0["High"]) + [m * 1.09]
    df = ohlc_from_close(closes, lows=lows, highs=highs)
    sigs = detect_signals(df, loose_params())
    assert sigs == []


def test_falling_ma60_rejected() -> None:
    """下降 MA60 是壓力，回踩不進。"""
    closes = list(np.linspace(110.0, 100.0, 80)) + [103.0, 106.0, 108.0, 104.0]
    df0 = ohlc_from_close(closes)
    ma = sma(df0["Close"].to_numpy(float), 60)
    m = float(ma[-1])
    closes = closes + [m * 1.004]
    lows = list(df0["Low"]) + [m * 0.998]
    highs = list(df0["High"]) + [m * 1.01]
    df = ohlc_from_close(closes, lows=lows, highs=highs)
    funnel: dict = {}
    sigs = detect_signals(df, loose_params(), funnel=funnel)
    assert sigs == []


def test_simulate_stop_and_target() -> None:
    base = rising_base(80)
    ext = [102.0, 105.0, 108.0, 107.0, 105.5]
    df0 = ohlc_from_close(base + ext)
    ma = sma(df0["Close"].to_numpy(float), 60)
    m = float(ma[-1])
    entry_px = m * 1.008
    stop_low = m * 0.997
    # 先做一筆未平，讀出實際停損價再打穿它
    probe = ohlc_from_close(
        base + ext + [entry_px],
        lows=list(df0["Low"]) + [stop_low],
        highs=list(df0["High"]) + [entry_px + 0.3],
    )
    sigs = detect_signals(probe, loose_params())
    assert len(sigs) == 1
    planned_stop = simulate(probe, sigs, loose_params(time_bars=8))[0].stop_price

    closes = base + ext + [entry_px, planned_stop - 0.5]
    lows = list(df0["Low"]) + [stop_low, planned_stop - 0.6]
    highs = list(df0["High"]) + [entry_px + 0.3, planned_stop]
    df = ohlc_from_close(closes, lows=lows, highs=highs)
    sigs = detect_signals(df, loose_params())
    trades = simulate(df, sigs, loose_params(time_bars=8))
    assert trades[0].exit_reason == "stop"
    assert trades[0].pnl_points < 0

    target = entry_px + 2 * (entry_px - planned_stop)
    closes2 = base + ext + [entry_px, target + 1]
    lows2 = list(df0["Low"]) + [stop_low, entry_px]
    highs2 = list(df0["High"]) + [entry_px + 0.3, target + 1]
    df2 = ohlc_from_close(closes2, lows=lows2, highs=highs2)
    sigs2 = detect_signals(df2, loose_params())
    trades2 = simulate(df2, sigs2, loose_params(time_bars=8))
    assert trades2[0].exit_reason == "target"
    assert trades2[0].pnl_points > 0


def test_filter_entry_window() -> None:
    base = rising_base(80)
    ext = [102.0, 105.0, 108.0, 107.0, 105.5]
    df0 = ohlc_from_close(base + ext)
    ma = sma(df0["Close"].to_numpy(float), 60)
    m = float(ma[-1])
    closes = base + ext + [m * 1.008]
    lows = list(df0["Low"]) + [m * 0.997]
    highs = list(df0["High"]) + [m * 1.012]
    df = ohlc_from_close(closes, lows=lows, highs=highs)
    sigs = detect_signals(df, loose_params())
    assert sigs
    assert len(filter_entry_window(df, sigs, 14)) == 1
    assert filter_entry_window(df, sigs, 0) == list(sigs)


def test_summarize_empty() -> None:
    stats = summarize_trades([])
    assert stats["count"] == 0
    assert stats["win_rate"] == 0.0


def main() -> int:
    test_sma()
    test_breakout_then_ma60_retest_enters()
    test_no_extension_no_signal()
    test_breakdown_is_not_retest()
    test_reclaim_from_below_is_not_this_setup()
    test_chase_extended_close_rejected()
    test_first_retest_only()
    test_same_bar_breakout_wick_not_entry()
    test_falling_ma60_rejected()
    test_simulate_stop_and_target()
    test_filter_entry_window()
    test_summarize_empty()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

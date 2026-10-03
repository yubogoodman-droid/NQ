#!/usr/bin/env python3
"""Synthetic tests for 晶心科型 1h 突破 MA60 後回測（不打 Yahoo）。"""

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
    resolve_pool,
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


def andes_shape() -> pd.DataFrame:
    """70 根在 100、12 根跌到 92、站回 101、拉開 108、再踩回均線。"""
    base = [100.0] * 70
    below = [98.0, 97.0, 96.0, 95.0, 94.0, 93.5, 93.0, 92.5, 92.0, 92.2, 92.6, 93.0]
    cross = [101.0]
    ext = [103.0, 106.0, 108.0, 107.0]
    df0 = ohlc_from_close(base + below + cross + ext)
    ma = sma(df0["Close"].to_numpy(float), 60)
    m = float(ma[-1])
    pb_c = m * 1.008
    pb_l = m * 0.997
    closes = base + below + cross + ext + [pb_c]
    lows = list(df0["Low"]) + [pb_l]
    highs = list(df0["High"]) + [max(pb_c, m * 1.012)]
    return ohlc_from_close(closes, lows=lows, highs=highs)


def test_sma() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
    ma = sma(x, 3)
    assert np.isnan(ma[1])
    assert abs(ma[2] - 2.0) < 1e-9


def test_andes_cross_then_retest_enters() -> None:
    df = andes_shape()
    funnel: dict = {}
    sigs = detect_signals(df, loose_params(), funnel=funnel)
    assert len(sigs) == 1, funnel
    s = sigs[0]
    assert s.entry_idx == len(df) - 1
    assert s.bars_below >= 12
    assert s.depth_pct >= 0.035
    assert s.ext_pct >= 0.03
    assert s.entry_idx > s.peak_idx
    assert s.entry_price <= s.ma60 * 1.025


def test_already_above_without_cross_rejected() -> None:
    """一直在均線上的右上角回踩，不通知。"""
    closes = list(np.linspace(96.0, 108.0, 80)) + [107.0, 106.0, 105.0]
    df = ohlc_from_close(closes)
    sigs = detect_signals(df, loose_params())
    assert sigs == []


def test_short_dip_rejected() -> None:
    """上升段只跌 2 根再站上，是假跌，不要。"""
    base = [100.0] * 70
    dip = [98.5, 98.0]
    cross = [101.0, 105.0, 108.0]
    df0 = ohlc_from_close(base + dip + cross)
    ma = sma(df0["Close"].to_numpy(float), 60)
    m = float(ma[-1])
    df = ohlc_from_close(
        base + dip + cross + [m * 1.006],
        lows=list(df0["Low"]) + [m * 0.998],
        highs=list(df0["High"]) + [m * 1.01],
    )
    funnel: dict = {}
    sigs = detect_signals(df, loose_params(), funnel=funnel)
    assert sigs == []
    assert funnel.get("too_short", 0) >= 1


def test_cross_already_extended_rejected() -> None:
    """站上那根已經離開均線很多，是右上角噴出，不從這裡起算。"""
    base = [100.0] * 70
    below = [97.0, 96.0, 95.0, 94.0, 93.5, 93.0, 92.5, 92.0, 92.2, 92.5, 92.8, 93.0]
    # 一根跳到 +8%
    cross = [108.0, 109.0, 107.0]
    df0 = ohlc_from_close(base + below + cross)
    ma = sma(df0["Close"].to_numpy(float), 60)
    m = float(ma[-1])
    df = ohlc_from_close(
        base + below + cross + [m * 1.006],
        lows=list(df0["Low"]) + [m * 0.998],
        highs=list(df0["High"]) + [m * 1.01],
    )
    funnel: dict = {}
    sigs = detect_signals(df, loose_params(), funnel=funnel)
    assert sigs == []
    assert funnel.get("cross_extended", 0) >= 1


def test_chase_extended_close_rejected() -> None:
    df = andes_shape()
    # 把回踩那根收盤改成遠離 MA60
    closes = list(df["Close"])
    ma = sma(df["Close"].to_numpy(float), 60)
    closes[-1] = float(ma[-2]) * 1.08
    lows = list(df["Low"])
    highs = list(df["High"])
    highs[-1] = closes[-1] + 0.3
    df2 = ohlc_from_close(closes, lows=lows, highs=highs)
    sigs = detect_signals(df2, loose_params())
    assert sigs == []


def test_first_retest_only() -> None:
    df = andes_shape()
    s = detect_signals(df, loose_params())[0]
    extra_c = list(df["Close"]) + [s.ma60 * 1.02, s.ma60 * 1.006]
    extra_l = list(df["Low"]) + [s.ma60 * 1.01, s.ma60 * 0.997]
    extra_h = list(df["High"]) + [s.ma60 * 1.03, s.ma60 * 1.012]
    df2 = ohlc_from_close(extra_c, lows=extra_l, highs=extra_h)
    sigs = detect_signals(df2, loose_params())
    assert len(sigs) == 1


def test_falling_ma60_still_ok() -> None:
    """晶心科回測時 MA60 還在微降，仍然要進。"""
    df = andes_shape()
    sigs = detect_signals(df, loose_params(require_rising_ma=False))
    assert len(sigs) == 1


def test_simulate_stop_and_target() -> None:
    df = andes_shape()
    sigs = detect_signals(df, loose_params())
    assert len(sigs) == 1
    planned = simulate(df, sigs, loose_params(time_bars=8))[0].stop_price
    closes = list(df["Close"]) + [planned - 0.5]
    lows = list(df["Low"]) + [planned - 0.6]
    highs = list(df["High"]) + [planned]
    df_stop = ohlc_from_close(closes, lows=lows, highs=highs)
    trades = simulate(df_stop, detect_signals(df_stop, loose_params()), loose_params(time_bars=8))
    assert trades[0].exit_reason == "stop"

    entry = sigs[0].entry_price
    target = entry + 2 * (entry - planned)
    closes2 = list(df["Close"]) + [target + 1]
    lows2 = list(df["Low"]) + [entry]
    highs2 = list(df["High"]) + [target + 1]
    df_tp = ohlc_from_close(closes2, lows=lows2, highs=highs2)
    trades2 = simulate(df_tp, detect_signals(df_tp, loose_params()), loose_params(time_bars=8))
    assert trades2[0].exit_reason == "target"


def test_filter_entry_window() -> None:
    df = andes_shape()
    sigs = detect_signals(df, loose_params())
    assert sigs
    assert len(filter_entry_window(df, sigs, 14)) == 1
    assert filter_entry_window(df, sigs, 0) == list(sigs)


def test_summarize_empty() -> None:
    stats = summarize_trades([])
    assert stats["count"] == 0


def test_resolve_pool_does_not_refill() -> None:
    assert resolve_pool(200, 0) == 200
    assert resolve_pool(200, 200) == 200
    assert resolve_pool(200, 400) == 400
    assert resolve_pool(0, 0) == 0


def main() -> int:
    test_sma()
    test_andes_cross_then_retest_enters()
    test_already_above_without_cross_rejected()
    test_short_dip_rejected()
    test_cross_already_extended_rejected()
    test_chase_extended_close_rejected()
    test_first_retest_only()
    test_falling_ma60_still_ok()
    test_simulate_stop_and_target()
    test_filter_entry_window()
    test_summarize_empty()
    test_resolve_pool_does_not_refill()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

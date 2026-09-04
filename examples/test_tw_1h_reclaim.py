#!/usr/bin/env python3
"""Synthetic tests for 台股 1h 破底翻（寬鬆版，不打 Yahoo）。"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tw_1h_reclaim import (  # noqa: E402
    TPE,
    detect_signals,
    filter_entry_window,
    loose_params,
    simulate,
    sma,
    strict_params,
    summarize_trades,
)

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
        lows = closes - 0.4
    if highs is None:
        highs = np.maximum(closes, np.asarray(lows, dtype=float)) + 0.3
    opens = np.concatenate([[closes[0]], closes[:-1]])
    df = pd.DataFrame(
        {
            "Open": opens,
            "High": np.asarray(highs, dtype=float),
            "Low": np.asarray(lows, dtype=float),
            "Close": closes,
            "Volume": np.full(n, 1000.0),
        },
        index=session_index(n),
    )
    return df


def valid_closes():
    """30 根在 100 上，7 根跌到 96，再站回 101。"""
    warmup = [100.0] * 30
    below = [98.6, 97.8, 96.9, 96.2, 96.0, 96.4, 96.9]
    reclaim = [101.0, 101.4, 101.8]
    closes = warmup + below + reclaim
    lows = [c - 0.3 for c in warmup] + [98.2, 97.4, 96.5, 95.8, 95.4, 96.0, 96.5] + [100.4, 100.8, 101.2]
    return closes, lows


def test_sma() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
    ma = sma(x, 3)
    assert np.isnan(ma[1])
    assert abs(ma[2] - 2.0) < 1e-9
    assert abs(ma[4] - 4.0) < 1e-9


def test_valid_wave_and_entry() -> None:
    closes, lows = valid_closes()
    df = ohlc_from_close(closes, lows=lows)
    funnel: dict = {}
    sigs = detect_signals(df, loose_params(), funnel=funnel)
    assert funnel.get("wave_ok") == 1
    assert len(sigs) == 1
    s = sigs[0]
    assert s.wave.bars_below == 7
    assert s.wave.fake_stands == 0
    assert s.wave.depth_pct >= 0.018
    assert s.entry_idx == s.wave.reclaim_idx
    assert s.entry_price > s.ma5
    assert s.entry_price > s.ma10
    assert s.entry_price > s.ma20
    # 破底那根低點 95.4 必須低於前面 16 根
    assert abs(s.wave.trough_low - 95.4) < 1e-9


def test_fake_stands_still_count() -> None:
    warmup = [100.0] * 30
    # 3 根下面 → 1 根假站上 → 5 根再下去 → 站回
    body = [98.5, 97.8, 97.2, 100.2, 97.0, 96.4, 95.9, 96.2, 96.6]
    lows = [c - 0.3 for c in warmup] + [98.1, 97.4, 96.8, 99.6, 96.6, 96.0, 95.3, 95.8, 96.2]
    reclaim = [101.0]
    df = ohlc_from_close(warmup + body + reclaim, lows=lows + [100.4])
    sigs = detect_signals(df, loose_params())
    assert len(sigs) == 1
    assert sigs[0].wave.fake_stands == 1
    assert sigs[0].wave.bars_below >= 4


def test_three_fakes_rejected() -> None:
    warmup = [100.0] * 30
    # 三段淺跌 + 假站上。深度不夠所以中間站上都不算成波，第三根假站結束。
    body = [99.6, 99.5, 100.3, 99.6, 99.4, 100.2, 99.5, 99.4, 100.4]
    lows = [c - 0.15 for c in (warmup + body)]
    df = ohlc_from_close(warmup + body, lows=lows)
    funnel: dict = {}
    sigs = detect_signals(df, loose_params(), funnel=funnel)
    assert sigs == []
    assert funnel.get("too_many_fakes", 0) + funnel.get("shallow", 0) + funnel.get("too_short", 0) >= 1


def test_too_short_rejected() -> None:
    warmup = [100.0] * 30
    below = [98.0, 96.5]  # 只待 2 根
    reclaim = [101.0, 101.2, 101.4]
    lows = [c - 0.3 for c in warmup] + [97.6, 95.2] + [100.4, 100.8, 101.0]
    df = ohlc_from_close(warmup + below + reclaim, lows=lows)
    funnel: dict = {}
    sigs = detect_signals(df, loose_params(), funnel=funnel)
    assert sigs == []
    assert funnel.get("too_short", 0) >= 1


def test_shallow_rejected() -> None:
    warmup = [100.0] * 30
    # 深度約 1.0% < 1.8%
    below = [99.4, 99.3, 99.2, 99.2, 99.3, 99.4]
    lows = [c - 0.15 for c in warmup] + [99.1, 99.0, 98.95, 98.9, 99.0, 99.1]
    reclaim = [100.4, 100.6, 100.8]
    df = ohlc_from_close(warmup + below + reclaim, lows=lows + [100.1, 100.3, 100.5])
    funnel: dict = {}
    sigs = detect_signals(df, loose_params(), funnel=funnel)
    assert sigs == []
    assert funnel.get("shallow", 0) >= 1


def test_higher_low_w_rejected() -> None:
    warmup = [100.0] * 30
    lows_w = [c - 0.3 for c in warmup]
    lows_w[20] = 90.0  # 16 根視窗內已有更低點
    below = [98.6, 97.8, 96.9, 96.2, 96.0, 96.4, 96.9]
    below_lows = [98.2, 97.4, 96.5, 95.8, 95.4, 96.0, 96.5]
    reclaim = [101.0, 101.4, 101.6]
    df = ohlc_from_close(warmup + below + reclaim, lows=lows_w + below_lows + [100.4, 100.8, 101.0])
    funnel: dict = {}
    sigs = detect_signals(df, loose_params(), funnel=funnel)
    assert sigs == []
    assert funnel.get("not_16h_low", 0) >= 1


def test_entry_must_be_within_36_of_trough() -> None:
    closes, lows = valid_closes()
    df = ohlc_from_close(closes, lows=lows)
    # 破底在 below 第 5 根，站回在 7 根之後 → 間隔 3，entry_window=2 就來不及
    sigs = detect_signals(df, loose_params(entry_window=2))
    assert sigs == []
    sigs_ok = detect_signals(df, loose_params(entry_window=8))
    assert len(sigs_ok) == 1


def test_strict_rejects_fake_stand() -> None:
    warmup = [100.0] * 30
    body = [98.5, 97.8, 97.2, 100.2, 97.0, 96.4, 95.9, 96.2, 96.6]
    lows = [c - 0.3 for c in warmup] + [98.1, 97.4, 96.8, 99.6, 96.6, 96.0, 95.3, 95.8, 96.2]
    df = ohlc_from_close(warmup + body + [101.0], lows=lows + [100.4])
    assert detect_signals(df, loose_params())
    assert detect_signals(df, strict_params()) == []


def test_simulate_stop_and_target() -> None:
    closes, lows = valid_closes()
    # 進場後再做一根長下影打到破底
    closes = list(closes) + [96.0, 96.0]
    lows = list(lows) + [95.3, 95.5]
    df = ohlc_from_close(closes, lows=lows)
    sigs = detect_signals(df, loose_params())
    assert len(sigs) == 1
    trades = simulate(df, sigs, loose_params(time_bars=8))
    assert len(trades) == 1
    assert trades[0].exit_reason == "stop"
    assert trades[0].pnl_points < 0

    closes2, lows2 = valid_closes()
    # 進場後大漲打到 2R
    entry_px = 101.0
    stop = 95.4
    target = entry_px + 2 * (entry_px - stop)
    closes2 = list(closes2) + [target + 1]
    lows2 = list(lows2) + [101.0]
    highs2 = [c + 0.3 for c in closes2]
    highs2[-1] = target + 1
    df2 = ohlc_from_close(closes2, lows=lows2, highs=highs2)
    sigs2 = detect_signals(df2, loose_params())
    trades2 = simulate(df2, sigs2, loose_params(time_bars=8))
    assert trades2[0].exit_reason == "target"
    assert trades2[0].pnl_points > 0


def test_filter_entry_window() -> None:
    closes, lows = valid_closes()
    df = ohlc_from_close(closes, lows=lows)
    sigs = detect_signals(df, loose_params())
    assert sigs
    # 資料最後一根是進場後幾根，14 天一定涵蓋
    kept = filter_entry_window(df, sigs, 14)
    assert len(kept) == 1
    # days<=0 不過濾
    assert filter_entry_window(df, sigs, 0) == list(sigs)


def test_price_cap_keeps_cheap_entry() -> None:
    closes, lows = valid_closes()
    df = ohlc_from_close(closes, lows=lows)
    sigs = detect_signals(df, loose_params())
    assert sigs
    assert all(s.entry_price < 1000 for s in sigs)
    assert [s for s in sigs if s.entry_price >= 50] == sigs


def test_summarize() -> None:
    stats = summarize_trades([])
    assert stats["count"] == 0
    assert stats["win_rate"] == 0.0


def main() -> int:
    test_sma()
    test_valid_wave_and_entry()
    test_fake_stands_still_count()
    test_three_fakes_rejected()
    test_too_short_rejected()
    test_shallow_rejected()
    test_higher_low_w_rejected()
    test_entry_must_be_within_36_of_trough()
    test_strict_rejects_fake_stand()
    test_simulate_stop_and_target()
    test_filter_entry_window()
    test_price_cap_keeps_cheap_entry()
    test_summarize()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

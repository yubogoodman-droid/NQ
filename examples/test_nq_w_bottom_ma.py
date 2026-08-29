#!/usr/bin/env python3
"""Synthetic tests for NQ 低檔 W 底站上 MA60 做多（M 頭空單鏡像，不連 Yahoo）。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nq.patterns import WBottomPattern, detect_classic_w_bottoms  # noqa: E402
from nq_m_head import ET, TRAIL_STEPS_5M  # noqa: E402
from nq_w_bottom_ma import (  # noqa: E402
    Signal,
    TradeResult,
    apply_long_preset,
    generate_signals,
    run_backtest,
    run_tf_backtest,
    write_html_report,
)


def _make_w_bottom_bars(n: int = 320) -> pd.DataFrame:
    """慢跌後末端急殺，在低檔做出夠深的雙底，再快速站上 MA60。"""
    close = np.zeros(n, dtype=float)
    close[0] = 20000.0
    l1, peak, l2 = 200, 218, 236

    for i in range(1, 165):
        close[i] = close[i - 1] - 0.35
    for i in range(165, l1):
        close[i] = close[i - 1] - 3.40
    trough = close[l1 - 1] - 8.0
    close[l1] = trough
    bounce = 42.0
    steps_up = peak - l1
    for i in range(l1 + 1, peak + 1):
        close[i] = trough + bounce * (i - l1) / steps_up
    steps_down = l2 - peak
    for i in range(peak + 1, l2):
        close[i] = close[peak] - (close[peak] - trough) * (i - peak) / steps_down
    close[l2] = trough
    for i in range(l2 + 1, n):
        close[i] = close[i - 1] + 8.0

    high = close + 1.0
    low = close - 1.0
    trough_low = trough - 4.0
    low[l1] = trough_low
    low[l2] = trough_low
    for i in list(range(l1 - 8, l1)) + list(range(l1 + 1, l1 + 9)):
        if 0 <= i < n:
            low[i] = max(float(low[i]), trough_low + 8.0)
    for i in list(range(l2 - 8, l2)) + list(range(l2 + 1, l2 + 9)):
        if 0 <= i < n:
            low[i] = max(float(low[i]), trough_low + 8.0)
    high[peak] = close[peak] + 3.0

    idx = pd.date_range("2026-08-17 09:30", periods=n, freq="1min", tz=ET)
    return pd.DataFrame(
        {
            "open": np.r_[close[0], close[:-1]],
            "high": high,
            "low": low,
            "close": close,
            "volume": np.full(n, 80.0),
        },
        index=idx,
    )


def test_detect_classic_w_geometry() -> None:
    df = _make_w_bottom_bars()
    patterns = detect_classic_w_bottoms(df)
    assert patterns, "synthetic W-bottom should be detected"
    p = patterns[0]
    assert p.first_low_idx < p.second_low_idx
    assert p.second_low >= p.first_low - 0.25
    assert abs(p.first_low - p.second_low) / p.trough < 0.002
    assert p.neckline > max(p.first_low, p.second_low)


def test_detect_and_simulate_long() -> None:
    df = _make_w_bottom_bars()
    funnel: dict = {}
    sigs = generate_signals(df, funnel=funnel)
    assert sigs, f"expected a long after MA60 reclaim, funnel={funnel}"
    sig = sigs[0]
    assert sig.entry > sig.stop_loss
    assert sig.target > sig.entry
    assert sig.entry > sig.ma60 - 1e-9
    assert sig.entry > sig.pattern.neckline
    assert sig.bar_idx > sig.pattern.second_low_idx
    assert sig.ribbon_spread >= 28.0

    trades = run_backtest(df, sigs, max_bars_hold=80)
    assert trades
    assert isinstance(trades[0], TradeResult)
    assert trades[0].exit_idx >= trades[0].signal.bar_idx
    assert trades[0].pnl_points == trades[0].exit_price - trades[0].signal.entry


def test_reject_lower_low() -> None:
    """右底再破底是下跌中繼，不該當成低檔 W 底。"""
    df = _make_w_bottom_bars()
    low = df["low"].to_numpy(copy=True)
    l2 = 236
    low[l2] = float(low[200]) - 8.0
    df = df.copy()
    df["low"] = low
    patterns = detect_classic_w_bottoms(df)
    assert all(p.second_low_idx != l2 for p in patterns)


def test_no_signal_without_ma60_reclaim() -> None:
    df = _make_w_bottom_bars()
    close = df["close"].to_numpy(copy=True)
    l2 = 236
    hold = float(close[l2])
    for i in range(l2 + 1, len(close)):
        close[i] = hold
    df = df.copy()
    df["close"] = close
    high = df["high"].to_numpy(copy=True)
    low = df["low"].to_numpy(copy=True)
    for i in range(l2 + 1, len(close)):
        high[i] = hold + 0.8
        low[i] = hold - 0.8
    df["high"] = high
    df["low"] = low
    funnel: dict = {}
    sigs = generate_signals(df, funnel=funnel)
    assert not sigs, f"should not enter without MA60 reclaim, got {len(sigs)} funnel={funnel}"


def test_skip_tangled_ribbon() -> None:
    df = _make_w_bottom_bars()
    funnel: dict = {}
    sigs = generate_signals(df, funnel=funnel, min_ribbon_spread=10_000.0)
    assert not sigs, f"impossible spread should skip, got {len(sigs)} funnel={funnel}"
    assert funnel.get("skip_tangled", 0) >= 1


def test_trail_locks_after_rally() -> None:
    """浮盈見過 1.6R 後鎖 1.2R；回撤打到鎖利就出場。"""
    n = 24
    entry_idx = 6
    entry, risk = 10000.0, 100.0
    stop, target = 9900.0, 10200.0
    close = np.full(n, entry)
    high = close + 2.0
    low = close - 2.0
    close[entry_idx + 1] = entry + 80.0
    high[entry_idx + 1] = entry + 80.0
    low[entry_idx + 1] = entry + 60.0
    close[entry_idx + 2] = entry + 160.0
    high[entry_idx + 2] = entry + 160.0
    low[entry_idx + 2] = entry + 150.0
    close[entry_idx + 3] = entry + 110.0
    high[entry_idx + 3] = entry + 130.0
    low[entry_idx + 3] = entry + 105.0
    idx = pd.date_range("2026-08-25 19:00", periods=n, freq="5min", tz=ET)
    df = pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": 50.0},
        index=idx,
    )
    pattern = WBottomPattern(1, 3, 2, stop + 8.0, stop + 8.0, entry - 20.0)
    sig = Signal(
        timestamp=df.index[entry_idx],
        entry=entry,
        stop_loss=stop,
        target=target,
        pattern=pattern,
        bar_idx=entry_idx,
        ma60=entry - 30.0,
        ma20=entry - 15.0,
        ma5=entry - 5.0,
    )
    trades = run_backtest(df, [sig], max_bars_hold=12)
    assert trades
    t = trades[0]
    assert t.exit_reason == "trail_stop", t
    assert abs(t.pnl_points - 120.0) < 0.26


def test_trail_does_not_block_two_r() -> None:
    n = 20
    entry_idx = 4
    entry, risk = 10000.0, 100.0
    stop, target = 9900.0, 10200.0
    close = np.full(n, entry)
    high = close + 1.0
    low = close - 1.0
    close[entry_idx + 1] = entry + 210.0
    high[entry_idx + 1] = entry + 210.0
    low[entry_idx + 1] = entry + 190.0
    idx = pd.date_range("2026-08-21 09:00", periods=n, freq="5min", tz=ET)
    df = pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": 50.0},
        index=idx,
    )
    pattern = WBottomPattern(1, 2, 1, stop + 8.0, stop + 8.0, entry - 20.0)
    sig = Signal(
        timestamp=df.index[entry_idx],
        entry=entry,
        stop_loss=stop,
        target=target,
        pattern=pattern,
        bar_idx=entry_idx,
        ma60=entry - 30.0,
        ma20=entry - 15.0,
        ma5=entry - 5.0,
    )
    trades = run_backtest(df, [sig], max_bars_hold=10)
    assert trades[0].exit_reason == "take_profit"
    assert abs(trades[0].pnl_points - 200.0) < 1e-9


def test_five_m_locks_one_r_giveback() -> None:
    n = 20
    entry_idx = 4
    entry, risk = 10000.0, 100.0
    stop, target = 9900.0, 10200.0
    close = np.full(n, entry)
    high = close + 2.0
    low = close - 2.0
    close[entry_idx + 1] = entry + 105.0
    high[entry_idx + 1] = entry + 105.0
    low[entry_idx + 1] = entry + 90.0
    close[entry_idx + 2] = entry + 50.0
    high[entry_idx + 2] = entry + 80.0
    low[entry_idx + 2] = entry + 40.0
    idx = pd.date_range("2026-08-05 03:30", periods=n, freq="5min", tz=ET)
    df = pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": 50.0},
        index=idx,
    )
    pattern = WBottomPattern(1, 2, 1, stop + 8.0, stop + 8.0, entry - 20.0)
    sig = Signal(
        timestamp=df.index[entry_idx],
        entry=entry,
        stop_loss=stop,
        target=target,
        pattern=pattern,
        bar_idx=entry_idx,
        ma60=entry - 30.0,
        ma20=entry - 15.0,
        ma5=entry - 5.0,
        timeframe="5m",
    )
    trades = run_backtest(df, [sig], max_bars_hold=10, trail_steps=TRAIL_STEPS_5M)
    assert trades[0].exit_reason == "trail_stop"
    assert abs(trades[0].pnl_points - 50.0) < 0.26


def test_skip_slow_sandwich_still_allows_rally() -> None:
    df = _make_w_bottom_bars()
    funnel: dict = {}
    sigs = generate_signals(df, funnel=funnel, skip_slow_sandwich=True)
    assert sigs, f"clean rally should still enter, funnel={funnel}"


def test_long_preset_maps_m_head_windows() -> None:
    p1 = apply_long_preset("1m")
    p5 = apply_long_preset("5m")
    assert p1["min_bars_between_lows"] == 20
    assert p5["min_bars_between_lows"] == 4
    assert p5["low_level_lookback"] == 24
    assert p5["stop_buffer"] == 36.0
    assert p5["skip_slow_sandwich"] is True
    assert "skip_before_minutes" not in p5


def test_5m_preset_still_fires() -> None:
    df = _make_w_bottom_bars()
    _, trades, funnel = run_tf_backtest(df, "5m")
    assert trades, f"5m preset should still catch the synthetic rally, funnel={funnel}"
    assert trades[0].signal.timeframe == "5m"


def test_write_html_report() -> None:
    df = _make_w_bottom_bars()
    sigs = generate_signals(df)
    trades = run_backtest(df, sigs, max_bars_hold=80)
    out = Path("/tmp/nq_w_bottom_ma_test.html")
    path = write_html_report(out, df, trades, "NQ=F", "demo")
    text = path.read_text(encoding="utf-8")
    assert "低檔W底" in text
    if trades:
        assert "img/" in text
        img_dir = path.parent / "img"
        assert any(img_dir.glob("w01_*.png")), "expected a static trade PNG"


def main() -> int:
    test_detect_classic_w_geometry()
    test_detect_and_simulate_long()
    test_reject_lower_low()
    test_no_signal_without_ma60_reclaim()
    test_skip_tangled_ribbon()
    test_trail_locks_after_rally()
    test_trail_does_not_block_two_r()
    test_five_m_locks_one_r_giveback()
    test_skip_slow_sandwich_still_allows_rally()
    test_long_preset_maps_m_head_windows()
    test_5m_preset_still_fires()
    test_write_html_report()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

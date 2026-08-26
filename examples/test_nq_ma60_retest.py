#!/usr/bin/env python3
"""Synthetic tests for NQ 1m 破底後回踩 MA60 (no Yahoo)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nq_ma60_retest import (  # noqa: E402
    ET,
    MAX_BELOW_MA60,
    MIN_RETEST_GAP,
    TradeResult,
    detect_signals,
    quality_from_retest,
    simulate,
    write_html_report,
)
from nq_ma_reclaim import sma  # noqa: E402


def test_quality_from_retest() -> None:
    assert quality_from_retest(2.0, True, 34.0) == (3, "A")
    assert quality_from_retest(2.0, True, 49.0) == (3, "A")
    assert quality_from_retest(2.0, True, 80.0) == (2, "A")
    assert quality_from_retest(-1.0, True, 34.0) == (2, "A")
    assert quality_from_retest(-1.0, False, 80.0) == (0, "C")


def _base_dump_recover(n: int = 340) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    """Chop, dump below the 2h low, then recover. Caller paints the retest."""
    close = np.zeros(n, dtype=float)
    close[0] = 20000.0
    for i in range(1, 80):
        close[i] = close[i - 1] + 0.55
    base = close[79]
    for i in range(80, 220):
        close[i] = base + (1.2 if i % 2 == 0 else -0.8)
    break_i = 220
    close[break_i] = base - 28.0
    for i, px in enumerate((base - 18.0, base - 8.0, base + 2.0, base + 14.0, base + 22.0)):
        close[break_i + 1 + i] = px
    for i in range(break_i + 6, n):
        close[i] = close[i - 1] + 1.6
    high = close + 1.4
    low = close - 1.4
    for i in range(80, 220):
        low[i] = min(close[i] - 0.4, base - 1.6)
        high[i] = close[i] + 0.4
    low[break_i] = close[break_i] - 0.6
    return close, high, low, base, break_i


def _to_df(close, high, low) -> pd.DataFrame:
    n = len(close)
    idx = pd.date_range("2026-08-17 11:00", periods=n, freq="1min", tz=ET)
    return pd.DataFrame(
        {
            "Open": np.r_[close[0], close[:-1]],
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": np.full(n, 80.0),
        },
        index=idx,
    )


def _make_retest_bars(n: int = 340) -> pd.DataFrame:
    """Chop to pin a 2h low, fake-break below MA60, reclaim, then retest MA60."""
    close, high, low, _base, break_i = _base_dump_recover(n)
    ma60 = sma(close, 60)
    # leave MA60, then come back to touch it
    retest_i = break_i + 16
    for i in range(break_i + 6, retest_i):
        if not np.isnan(ma60[i]):
            close[i] = float(ma60[i]) + 18.0
            high[i] = close[i] + 1.5
            low[i] = float(ma60[i]) + 10.0
    if retest_i < n and not np.isnan(ma60[retest_i]):
        low[retest_i] = float(ma60[retest_i]) - 1.0
        close[retest_i] = float(ma60[retest_i]) + 4.0
        high[retest_i] = close[retest_i] + 1.2
        for i in range(retest_i + 1, n):
            close[i] = close[i - 1] + 1.2
            high[i] = close[i] + 1.2
            low[i] = close[i] - 0.8
    return _to_df(close, high, low)


def _make_no_retest_bars(n: int = 340) -> pd.DataFrame:
    """Same dump + reclaim, but price never comes back to MA60."""
    close, high, low, _base, break_i = _base_dump_recover(n)
    ma60 = sma(close, 60)
    for i in range(break_i + 4, n):
        if np.isnan(ma60[i]):
            continue
        close[i] = max(float(close[i]), float(ma60[i]) + 22.0)
        low[i] = max(float(low[i]), float(ma60[i]) + 18.0)
        high[i] = max(float(high[i]), close[i] + 1.2)
    return _to_df(close, high, low)


def _make_july22_bars(n: int = 360) -> pd.DataFrame:
    """07-22：破底距 MA60 ≈49 點，翻上後很快回踩，再 V 型拉升。"""
    close, high, low, base, break_i = _base_dump_recover(n)
    close[break_i] = base - 49.0
    low[break_i] = close[break_i] - 0.8
    high[break_i] = close[break_i] + 1.0
    # climb back through MA60 over ~20 bars
    for i, px in enumerate((base - 36.0, base - 24.0, base - 12.0, base - 4.0, base + 2.0, base + 8.0)):
        close[break_i + 1 + i] = px
        high[break_i + 1 + i] = px + 1.2
        low[break_i + 1 + i] = px - 1.2
    ma60 = sma(close, 60)
    retest_i = break_i + 10
    for i in range(break_i + 7, retest_i):
        if not np.isnan(ma60[i]):
            close[i] = float(ma60[i]) + 10.0
            high[i] = close[i] + 1.4
            low[i] = float(ma60[i]) + 4.0
    if retest_i < n and not np.isnan(ma60[retest_i]):
        low[retest_i] = float(ma60[retest_i]) - 0.8
        close[retest_i] = float(ma60[retest_i]) + 5.0
        high[retest_i] = close[retest_i] + 1.5
        for i in range(retest_i + 1, n):
            close[i] = close[i - 1] + 2.4
            high[i] = close[i] + 1.4
            low[i] = close[i] - 0.6
    return _to_df(close, high, low)


def test_detect_and_simulate_retest() -> None:
    """08-25：貼季線破底（約 34 點）再回踩。"""
    df = _make_retest_bars()
    funnel: dict[str, int] = {}
    sigs = detect_signals(df, funnel=funnel, skip_hour_start=None, skip_hour_end=None)
    assert sigs, f"expected a retest signal, funnel={funnel}"
    sig = sigs[0]
    assert sig.entry_idx > sig.breakout_idx > sig.break_idx
    assert sig.entry_price > sig.stop_price
    assert sig.quality in {"A", "B", "C"}
    assert sig.entry_idx - sig.breakout_idx >= MIN_RETEST_GAP
    assert sig.entry_idx - sig.break_idx <= 60
    assert 0 < sig.below_ma60 <= MAX_BELOW_MA60
    assert sig.below_ma60 <= 45.0

    trades = simulate(df, sigs, preopen_flat=False, exit_on_ma60_lose=False)
    assert trades
    assert isinstance(trades[0], TradeResult)
    assert trades[0].exit_idx >= trades[0].entry_idx


def test_detect_july22_style_dump() -> None:
    """07-22：破底約 49 點、翻上後立刻回踩，也要進場。"""
    df = _make_july22_bars()
    funnel: dict[str, int] = {}
    sigs = detect_signals(df, funnel=funnel, skip_hour_start=None, skip_hour_end=None)
    assert sigs, f"expected july22-style signal, funnel={funnel}"
    sig = sigs[0]
    assert 45.0 < sig.below_ma60 <= MAX_BELOW_MA60
    assert sig.entry_idx > sig.breakout_idx > sig.break_idx
    assert sig.entry_idx - sig.breakout_idx >= MIN_RETEST_GAP


def test_no_signal_without_retest() -> None:
    df = _make_no_retest_bars()
    sigs = detect_signals(df, skip_hour_start=None, skip_hour_end=None)
    assert not sigs, "breakout without a MA60 retest should not enter"


def test_reject_break_too_far_from_ma60() -> None:
    """瀑布底離 MA60 太遠，不像截圖那種貼季線的破底。"""
    n = 340
    close, high, low, base, break_i = _base_dump_recover(n)
    close[break_i] = base - 80.0
    low[break_i] = close[break_i] - 0.6
    high[break_i] = close[break_i] + 1.0
    df = _to_df(close, high, low)
    funnel: dict[str, int] = {}
    sigs = detect_signals(df, funnel=funnel, skip_hour_start=None, skip_hour_end=None)
    assert not sigs, f"far 破底 should be skipped, funnel={funnel}"
    assert funnel.get("too_far", 0) >= 1 or funnel.get("break", 0) == 0


def test_write_html_report(tmp_path: Path | None = None) -> None:
    df = _make_retest_bars()
    sigs = detect_signals(df, skip_hour_start=None, skip_hour_end=None)
    trades = simulate(df, sigs, preopen_flat=False, exit_on_ma60_lose=False)
    out = Path("/tmp/nq_ma60_retest_test.html") if tmp_path is None else Path(tmp_path) / "r.html"
    path = write_html_report(out, df, trades, "NQ=F", "demo")
    text = path.read_text(encoding="utf-8")
    assert "回踩 MA60" in text
    if trades:
        assert "<img src='img/" in text
        img_dir = path.parent / "img"
        assert any(img_dir.glob("t01_*.png")), "expected a static trade PNG"
        assert any(img_dir.glob("t01_*_5m.png")), "expected a 5m reference PNG"
        assert "五分K對照" in text


def main() -> int:
    test_quality_from_retest()
    test_detect_and_simulate_retest()
    test_detect_july22_style_dump()
    test_no_signal_without_retest()
    test_reject_break_too_far_from_ma60()
    test_write_html_report()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

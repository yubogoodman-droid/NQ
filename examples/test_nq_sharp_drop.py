#!/usr/bin/env python3
"""Synthetic tests for NQ 5m 急跌 V 反 (no Yahoo)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nq_sharp_drop import (  # noqa: E402
    ET,
    TradeResult,
    _is_reversal_bar,
    detect_signals,
    quality_from_dump,
    simulate,
    summarize_trades,
    write_html_report,
)


def test_reversal_bar() -> None:
    assert _is_reversal_bar(100.0, 104.0, 98.0, 103.0)
    assert not _is_reversal_bar(104.0, 104.0, 98.0, 99.0)
    assert not _is_reversal_bar(100.0, 101.0, 99.0, 99.2)


def test_quality_from_dump() -> None:
    assert quality_from_dump(3.2, 2.1, True)[1] == "A"
    assert quality_from_dump(2.1, 1.5, False)[1] == "C"
    assert quality_from_dump(3.2, 1.2, False)[1] == "B"


def test_summarize_trades() -> None:
    class T:
        def __init__(self, pnl: float, quality: str = "A"):
            self.pnl_points = pnl
            self.quality = quality

    stats = summarize_trades([T(20.0, "A"), T(-8.0, "B")])
    assert stats["count"] == 2
    assert stats["wins"] == 1
    assert abs(stats["total_points"] - 12.0) < 1e-9


def _make_flush_bars(n: int = 180, bounce: bool = True) -> pd.DataFrame:
    """Ribbon grind, then a 6-bar flush through MA20, then a reversal bounce."""
    close = np.zeros(n, dtype=float)
    close[0] = 20000.0
    for i in range(1, 90):
        close[i] = close[i - 1] + (0.4 if i % 2 == 0 else -0.2)
    base = close[89]
    for i in range(90, 130):
        close[i] = base + (1.2 if i % 2 == 0 else -0.8)
    dump_i = 130
    close[dump_i] = base - 12.0
    close[dump_i + 1] = base - 28.0
    close[dump_i + 2] = base - 48.0
    close[dump_i + 3] = base - 70.0
    close[dump_i + 4] = base - 88.0
    if bounce:
        close[dump_i + 5] = base - 62.0
        close[dump_i + 6] = base - 48.0
        for i in range(dump_i + 7, n):
            close[i] = min(base + 4.0, close[i - 1] + 6.0)
    else:
        close[dump_i + 5] = base - 110.0
        for i in range(dump_i + 6, n):
            close[i] = close[i - 1] - 4.0

    high = close + 1.5
    low = close - 1.5
    open_ = np.r_[close[0], close[:-1]]
    for i in range(dump_i, dump_i + 5):
        open_[i] = close[i] + 8.0
        high[i] = open_[i] + 0.4
        low[i] = close[i] - 2.0
    if bounce:
        open_[dump_i + 5] = close[dump_i + 4]
        low[dump_i + 5] = close[dump_i + 4] - 1.0
        high[dump_i + 5] = close[dump_i + 5] + 1.0

    vol = np.full(n, 80.0)
    vol[dump_i : dump_i + 6] = 220.0

    idx = pd.date_range("2026-08-17 11:00", periods=n, freq="5min", tz=ET)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=idx,
    )


def test_detect_flush_bounce() -> None:
    df = _make_flush_bars(bounce=True)
    sigs = detect_signals(df, rth_only=False, skip_hour_start=None, skip_hour_end=None)
    assert sigs, "expected a MA20 reclaim after the synthetic dump"
    sig = sigs[0]
    assert sig.entry_idx > sig.dump_idx
    assert sig.entry_price > sig.stop_price
    assert sig.dump_low < sig.dump_high
    assert sig.drop_pts >= 40
    assert sig.entry_price > sig.ma20
    risk = sig.entry_price - sig.stop_price
    assert abs((sig.target_price - sig.entry_price) / risk - 1.5) < 1e-6


def test_no_signal_on_continued_dump() -> None:
    df = _make_flush_bars(bounce=False)
    sigs = detect_signals(df, rth_only=False, skip_hour_start=None, skip_hour_end=None)
    assert not sigs, "continued waterfall should not reclaim MA20"


def test_simulate_and_html(tmp_path: Path | None = None) -> None:
    df = _make_flush_bars(bounce=True)
    sigs = detect_signals(df, rth_only=False, skip_hour_start=None, skip_hour_end=None)
    trades = simulate(df, sigs)
    assert trades
    assert isinstance(trades[0], TradeResult)
    out = Path("/tmp/nq_sharp_drop_test.html") if tmp_path is None else Path(tmp_path) / "r.html"
    path = write_html_report(out, df, trades, "NQ=F", "demo")
    text = path.read_text(encoding="utf-8")
    assert "急跌" in text
    if trades:
        assert "<img src='img/" in text
        img_dir = path.parent / "img"
        assert any(img_dir.glob("t01_*.png")), "expected a static trade PNG"


def main() -> int:
    test_reversal_bar()
    test_quality_from_dump()
    test_summarize_trades()
    test_detect_flush_bounce()
    test_no_signal_on_continued_dump()
    test_simulate_and_html()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

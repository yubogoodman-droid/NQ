#!/usr/bin/env python3
"""Synthetic tests for NQ 5m 打底後 5/10/20 多排 (no Yahoo)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nq_5m_base_stack import (  # noqa: E402
    ET,
    TradeResult,
    detect_signals,
    quality_from_setup,
    simulate,
    sma,
    summarize_trades,
    write_html_report,
)


def test_quality_from_setup() -> None:
    assert quality_from_setup(152.0, 0.43, 16.8) == (3, "A")
    assert quality_from_setup(152.0, 0.43, 40.0) == (2, "A")
    assert quality_from_setup(90.0, 0.43, 40.0) == (1, "B")
    assert quality_from_setup(90.0, 0.80, 40.0) == (0, "C")


def test_sma() -> None:
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = sma(arr, 3)
    assert np.isnan(out[1])
    assert abs(out[2] - 2.0) < 1e-9
    assert abs(out[4] - 4.0) < 1e-9


def test_summarize_trades() -> None:
    class T:
        def __init__(self, pnl: float, quality: str = "A"):
            self.pnl_points = pnl
            self.quality = quality

    stats = summarize_trades([T(10.0, "A"), T(-4.0, "B"), T(2.0, "A")])
    assert stats["count"] == 3
    assert stats["wins"] == 2
    assert abs(stats["total_points"] - 8.0) < 1e-9
    assert stats["by_quality"]["A"]["n"] == 2


def _make_base_stack_bars(n: int = 220, bounce: bool = True) -> pd.DataFrame:
    """Grind, dump ~130 pts, chop a base, then lift until 5/10/20 stack."""
    close = np.zeros(n, dtype=float)
    close[0] = 20000.0
    for i in range(1, 90):
        close[i] = close[i - 1] + (0.35 if i % 2 == 0 else -0.15)
    peak = close[89]
    # dump 20 bars
    for k, i in enumerate(range(90, 110)):
        close[i] = peak - 6.5 * (k + 1)
    floor = close[109]
    # unique swing low a few bars into the base
    close[112] = floor - 8.0
    bottom = close[112]
    if bounce:
        for i in range(110, 112):
            close[i] = floor + (2.0 if i % 2 == 0 else -1.0)
        for i in range(113, 125):
            close[i] = bottom + 6.0 + (3.0 if i % 2 == 0 else 1.5)
        close[125] = bottom + 28.0
        for i in range(126, n):
            close[i] = min(peak + 20.0, close[i - 1] + 4.2)
    else:
        for i in range(110, 112):
            close[i] = floor - 2.0
        for i in range(113, n):
            close[i] = close[i - 1] - 5.0

    high = close + 1.8
    low = close - 1.8
    open_ = np.r_[close[0], close[:-1]]
    for i in range(90, 110):
        open_[i] = close[i] + 4.0
        high[i] = open_[i] + 0.5
        low[i] = close[i] - 2.2
    low[112] = close[112] - 1.0
    high[112] = close[112] + 2.0
    if bounce:
        for i in range(113, 125):
            open_[i] = close[i] - 1.0
            low[i] = min(close[i] - 1.2, bottom + 3.0)
            high[i] = close[i] + 1.5

    vol = np.full(n, 90.0)
    vol[90:113] = 200.0
    idx = pd.date_range("2026-08-25 18:00", periods=n, freq="5min", tz=ET)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=idx,
    )


def test_detect_base_then_stack() -> None:
    df = _make_base_stack_bars(bounce=True)
    sigs = detect_signals(df)
    assert sigs, "expected 5/10/20 stack after the synthetic base"
    sig = sigs[0]
    assert sig.entry_idx > sig.base_idx
    assert sig.entry_idx - sig.base_idx >= 6
    assert sig.entry_price > sig.stop_price
    assert sig.drop_pts >= 80
    assert sig.ma5 > sig.ma10 > sig.ma20
    assert sig.entry_price > sig.ma5
    risk = sig.entry_price - sig.stop_price
    assert (sig.target_price - sig.entry_price) / risk >= 1.99


def test_no_signal_on_continued_dump() -> None:
    df = _make_base_stack_bars(bounce=False)
    sigs = detect_signals(df)
    assert not sigs, "waterfall after the low should not count as 打底多排"


def test_simulate_and_html(tmp_path: Path | None = None) -> None:
    df = _make_base_stack_bars(bounce=True)
    sigs = detect_signals(df)
    trades = simulate(df, sigs)
    assert trades
    assert isinstance(trades[0], TradeResult)
    out = Path("/tmp/nq_5m_base_stack_test.html") if tmp_path is None else Path(tmp_path) / "r.html"
    path = write_html_report(out, df, trades, "NQ=F", "demo")
    text = path.read_text(encoding="utf-8")
    assert "打底" in text
    assert "多排" in text
    if trades:
        assert "<img src='img/" in text
        img_dir = path.parent / "img"
        assert any(img_dir.glob("t01_*.png")), "expected a static trade PNG"


def main() -> int:
    test_quality_from_setup()
    test_sma()
    test_summarize_trades()
    test_detect_base_then_stack()
    test_no_signal_on_continued_dump()
    test_simulate_and_html()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

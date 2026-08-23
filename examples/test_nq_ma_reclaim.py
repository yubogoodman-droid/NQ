#!/usr/bin/env python3
"""Synthetic tests for NQ 1m 破底翻 MA Reclaim (no Yahoo)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nq_ma_reclaim import (  # noqa: E402
    ET,
    TradeResult,
    detect_signals,
    quality_from_slopes,
    simulate,
    sma,
    summarize_trades,
    write_html_report,
)


def test_quality_from_slopes() -> None:
    assert quality_from_slopes(20.0, -10.0, -9.0) == (3, "A")
    assert quality_from_slopes(20.0, -10.0, 1.0) == (2, "A")
    assert quality_from_slopes(20.0, 0.0, 1.0) == (1, "B")
    assert quality_from_slopes(0.0, 0.0, float("nan")) == (0, "C")


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
    assert stats["by_quality"]["A"]["wins"] == 2


def _make_reclaim_bars(n: int = 280) -> pd.DataFrame:
    """Grind up, range to pin the 2h low, fake-break, then reclaim above MA20/30."""
    close = np.zeros(n, dtype=float)
    close[0] = 20000.0
    for i in range(1, 100):
        close[i] = close[i - 1] + 0.70
    base = close[99]
    for i in range(100, 220):
        close[i] = base + (2.0 if i % 2 == 0 else -1.0)
    break_i = 220
    close[break_i] = base - 18.0
    close[break_i + 1] = base + 6.0
    close[break_i + 2] = base + 24.0
    close[break_i + 3] = base + 32.0
    close[break_i + 4] = base + 40.0
    for i in range(break_i + 5, n):
        close[i] = close[i - 1] + 0.6
    high = close + 0.8
    low = close - 0.8
    for i in range(100, 220):
        low[i] = min(close[i] - 0.3, base - 1.5)
        high[i] = close[i] + 0.3
    low[break_i] = close[break_i] - 0.5
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


def test_detect_and_simulate_reclaim() -> None:
    df = _make_reclaim_bars()
    sigs = detect_signals(df)
    assert sigs, "expected at least one reclaim signal on the synthetic dump"
    sig = sigs[0]
    assert sig.entry_idx > sig.break_idx
    assert sig.entry_price > sig.stop_price
    assert sig.quality in {"A", "B", "C"}
    assert sig.entry_idx - sig.break_idx <= 15

    trades = simulate(df, sigs, preopen_flat=False, stop_on_m5_close=False)
    assert trades
    assert isinstance(trades[0], TradeResult)
    assert trades[0].exit_idx >= trades[0].entry_idx
    assert trades[0].pnl_points != 0 or trades[0].exit_reason


def test_write_html_report(tmp_path: Path | None = None) -> None:
    df = _make_reclaim_bars()
    sigs = detect_signals(df)
    trades = simulate(df, sigs, preopen_flat=False, stop_on_m5_close=False)
    out = Path("/tmp/nq_ma_reclaim_test.html") if tmp_path is None else Path(tmp_path) / "r.html"
    path = write_html_report(out, df, trades, "NQ=F", "demo")
    text = path.read_text(encoding="utf-8")
    assert "破底翻 MA Reclaim" in text
    assert ("entry" in text) or ("無交易" in text)


def main() -> int:
    test_quality_from_slopes()
    test_sma()
    test_summarize_trades()
    test_detect_and_simulate_reclaim()
    test_write_html_report()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

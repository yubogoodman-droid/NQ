#!/usr/bin/env python3
"""Synthetic tests for NQ 1m 破底翻 MA Reclaim (no Yahoo)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from types import SimpleNamespace

from nq_ma_reclaim import (  # noqa: E402
    ET,
    TradeResult,
    detect_kwargs,
    detect_signals,
    draw_event_png,
    parse_period_days,
    quality_from_slopes,
    simulate,
    sma,
    summarize_trades,
    write_html_report,
)


def test_hug_band_skips_flat_not_steep() -> None:
    """08-11 型走平要擋；07-27 21:25 型大跌收復（斜率 -12）不擋。"""
    kw = detect_kwargs(SimpleNamespace(reclaim_ma20_only=True))
    assert kw["require_ma30"] is False
    assert kw["min_ma20_slope"] == -13.0
    assert kw["hug_ma20_min_slope"] == -8.0
    assert -13.0 <= -12.12 < -8.0
    assert -8.0 <= -0.95 <= 0.5


def test_detect_kwargs_allow_open_hour() -> None:
    assert detect_kwargs(SimpleNamespace(loose=False)) == {}
    kw = detect_kwargs(SimpleNamespace(loose=False, allow_open_hour=True))
    assert kw["skip_hour_start"] is None
    assert kw["skip_hour_end"] is None
    kw20 = detect_kwargs(SimpleNamespace(loose=False, reclaim_ma20_only=True))
    assert kw20["require_ma30"] is False


def test_parse_period_days() -> None:
    assert parse_period_days("8d") == 8
    assert parse_period_days("30d") == 30
    assert parse_period_days("1mo") == 30
    assert parse_period_days("4w") == 28
    assert parse_period_days("5d") == 5


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


def test_trace_records_taken() -> None:
    df = _make_reclaim_bars()
    trace: list = []
    detect_signals(df, trace=trace)
    assert any(row["reason"] == "taken" for row in trace)


def test_draw_event_png() -> None:
    df = _make_reclaim_bars()
    path = draw_event_png(df, 222, Path("/tmp/nq_event_test.png"), "demo", break_idx=220)
    assert path.exists() and path.stat().st_size > 1000


def test_write_html_report(tmp_path: Path | None = None) -> None:
    df = _make_reclaim_bars()
    sigs = detect_signals(df)
    trades = simulate(df, sigs, preopen_flat=False, stop_on_m5_close=False)
    out = Path("/tmp/nq_ma_reclaim_test.html") if tmp_path is None else Path(tmp_path) / "r.html"
    path = write_html_report(out, df, trades, "NQ=F", "demo")
    text = path.read_text(encoding="utf-8")
    assert "破底翻 MA Reclaim" in text
    assert ("entry" in text) or ("無交易" in text)
    if trades:
        assert "<img src='img/" in text
        img_dir = path.parent / "img"
        assert any(img_dir.glob("t01_*.png")), "expected a static trade PNG"


def main() -> int:
    test_hug_band_skips_flat_not_steep()
    test_detect_kwargs_allow_open_hour()
    test_parse_period_days()
    test_quality_from_slopes()
    test_sma()
    test_summarize_trades()
    test_detect_and_simulate_reclaim()
    test_trace_records_taken()
    test_draw_event_png()
    test_write_html_report()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

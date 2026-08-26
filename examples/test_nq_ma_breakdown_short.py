#!/usr/bin/env python3
"""Synthetic tests for NQ 5m 同時跌破 MA5/10/20/30/60/120 做空."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nq.ma_breakdown_short import (  # noqa: E402
    MA_PERIODS,
    TradeResult,
    detect_signals,
    quality_from_break,
    simulate,
    sma,
    summarize_trades,
)
from nq_ma_breakdown_short import ET, make_demo_bars, write_html_report  # noqa: E402


def test_quality_from_break() -> None:
    assert quality_from_break(50.0, 25.0, True)[1] == "A"
    assert quality_from_break(10.0, 5.0, False)[1] == "C"


def test_summarize_trades() -> None:
    class T:
        def __init__(self, pnl: float, quality: str = "A"):
            self.pnl_points = pnl
            self.quality = quality

    stats = summarize_trades([T(20.0), T(-8.0)])
    assert stats["count"] == 2
    assert stats["wins"] == 1
    assert abs(stats["total_points"] - 12.0) < 1e-9


def _make_break_bars(*, dump: bool = True, only_ma5: bool = False, n: int = 180) -> pd.DataFrame:
    close = np.full(n, 20000.0)
    high = close + 3.0
    low = close - 3.0
    open_ = close.copy()
    i = 130
    if only_ma5:
        close[i] = 19996.0
        open_[i] = 20002.0
        high[i] = 20004.0
        low[i] = 19994.0
        for j in range(i + 1, n):
            close[j] = 19997.0
            open_[j] = 19998.0
            high[j] = 20000.0
            low[j] = 19995.0
    elif dump:
        close[i] = 19910.0
        open_[i] = 20002.0
        high[i] = 20008.0
        low[i] = 19898.0
        for j in range(i + 1, n):
            close[j] = close[j - 1] - 8.0
            open_[j] = close[j - 1]
            high[j] = open_[j] + 1.5
            low[j] = close[j] - 2.0
    idx = pd.date_range("2026-07-28 04:00", periods=n, freq="5min", tz=ET)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": np.full(n, 80.0)},
        index=idx,
    )


def test_detect_simultaneous_breakdown() -> None:
    df = _make_break_bars(dump=True)
    sigs = detect_signals(df)
    assert sigs, "expected a short when one bar closes under MA5/10/20/30/60/120"
    sig = sigs[0]
    close = df["Close"].to_numpy(float)
    assert close[sig.entry_idx] < sma(close, 5)[sig.entry_idx]
    assert close[sig.entry_idx] < sma(close, 120)[sig.entry_idx]
    assert sig.stop_price > sig.entry_price
    assert sig.target_price < sig.entry_price


def test_no_signal_if_only_ma5() -> None:
    df = _make_break_bars(only_ma5=True)
    sigs = detect_signals(df)
    assert not sigs, "a shallow dip under MA5 only should not short"


def test_no_signal_if_mas_break_one_by_one() -> None:
    """Uptrend ribbon, then grind down through MA5 first and MA120 last — not 同時."""
    n = 220
    close = 19000.0 + np.arange(n, dtype=float) * 4.0
    i0 = 170
    close[i0:] = close[i0 - 1] - np.arange(n - i0, dtype=float) * 6.0
    open_ = close.copy()
    open_[i0:] = close[i0:] + 4.0
    high = np.maximum(open_, close) + 2.0
    low = np.minimum(open_, close) - 2.0
    idx = pd.date_range("2026-07-28 04:00", periods=n, freq="5min", tz=ET)
    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": np.full(n, 80.0)},
        index=idx,
    )
    mas = {p: sma(close, p) for p in MA_PERIODS}
    first_all = None
    for i in range(max(MA_PERIODS), n):
        vals = [float(mas[p][i]) for p in MA_PERIODS]
        if any(np.isnan(v) for v in vals):
            continue
        if all(close[i] < v for v in vals):
            first_all = i
            break
    assert first_all is not None, "expected the grind to eventually close under all MAs"
    prev = [float(mas[p][first_all - 1]) for p in MA_PERIODS]
    assert not all(close[first_all - 1] >= v for v in prev), "precondition: prev bar already under some MAs"
    sigs = detect_signals(df, session_start=None, session_end=None)
    assert not sigs, "sequential ribbon breakdown must not count as 同時跌破"


def test_simulate_short_hits_target() -> None:
    df = _make_break_bars(dump=True)
    sigs = detect_signals(df)
    trades = simulate(df, sigs, max_hold=24)
    assert trades
    assert isinstance(trades[0], TradeResult)
    assert trades[0].pnl_points > 0
    assert trades[0].exit_reason in {"target", "stop", "time"}


def test_demo_html(tmp_path: Path | None = None) -> None:
    df = make_demo_bars()
    sigs = detect_signals(df)
    assert sigs
    trades = simulate(df, sigs)
    out = Path("/tmp/nq_ma_breakdown_short_test.html") if tmp_path is None else Path(tmp_path) / "r.html"
    path = write_html_report(out, df, trades, "NQ=F", "demo")
    text = path.read_text(encoding="utf-8")
    assert "跌破均線叢做空" in text
    assert "<img src='img/" in text


def main() -> int:
    test_quality_from_break()
    test_summarize_trades()
    test_detect_simultaneous_breakdown()
    test_no_signal_if_only_ma5()
    test_no_signal_if_mas_break_one_by_one()
    test_simulate_short_hits_target()
    test_demo_html()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Synthetic tests for NQ 1m 破4h低 → 站上 MA200（不打 Yahoo）。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nq_4h_ma200 import (  # noqa: E402
    ET,
    TradeResult,
    detect_signals,
    fmt_entry,
    parse_period_days,
    quality_from_setup,
    simulate,
    sma,
    summarize_trades,
    write_html_report,
)


def test_parse_period_days() -> None:
    assert parse_period_days("8d") == 8
    assert parse_period_days("30d") == 30
    assert parse_period_days("1mo") == 30


def test_quality_from_setup() -> None:
    assert quality_from_setup(37.0, 32, 7.0)[1] == "A"
    assert quality_from_setup(10.0, 50, 3.0)[1] == "C"
    assert quality_from_setup(25.0, 50, 3.0)[1] == "B"


def test_sma() -> None:
    out = sma(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 3)
    assert np.isnan(out[1])
    assert abs(out[2] - 2.0) < 1e-9


def _make_sep29_like(n: int = 420) -> pd.DataFrame:
    """Range, flush a 4h low, then climb back through MA200 with a 5/10/20/30 stack."""
    close = np.zeros(n, dtype=float)
    close[0] = 30440.0
    for i in range(1, 300):
        close[i] = 30440.0 + (3.0 if i % 2 == 0 else -2.0)
    # pin a 4h floor near 30410
    low = close - 2.0
    high = close + 2.0
    for i in range(80, 300):
        low[i] = min(close[i] - 1.0, 30410.0)
    break_i = 310
    close[break_i] = 30395.0
    low[break_i] = 30372.25
    high[break_i] = 30400.0
    close[break_i + 1] = 30382.0
    low[break_i + 1] = 30374.0
    high[break_i + 1] = 30390.0
    # grind back above MA200 (~30440 area after 200-bar average)
    px = 30390.0
    for i in range(break_i + 2, n):
        px += 4.2
        close[i] = px
        low[i] = px - 3.0
        high[i] = px + 3.0
    idx = pd.date_range("2026-09-29 00:00", periods=n, freq="1min", tz=ET)
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


def test_detects_sep29_like() -> None:
    df = _make_sep29_like()
    sigs = detect_signals(df)
    assert sigs, "expected a 4h-break → MA200 stack signal"
    sig = sigs[0]
    assert sig.entry_idx > sig.break_idx
    assert sig.bars_to_entry <= 60
    assert sig.entry_price > sig.ma200
    assert sig.ma5 > sig.ma10 > sig.ma20 > sig.ma30
    assert sig.break_low <= 30380.0


def test_no_signal_without_ma200() -> None:
    df = _make_sep29_like().copy()
    # flatten the bounce so price never recaptures MA200
    close = df["Close"].to_numpy(float).copy()
    close[312:] = 30390.0
    df["Close"] = close
    df["High"] = np.maximum(df["High"].to_numpy(float), close)
    df["Low"] = np.minimum(df["Low"].to_numpy(float), close)
    sigs = detect_signals(df)
    assert not sigs


def test_simulate_and_html(tmp_path: Path | None = None) -> None:
    df = _make_sep29_like()
    sigs = detect_signals(df)
    trades = simulate(df, sigs, preopen_flat=False)
    assert trades
    assert isinstance(trades[0], TradeResult)
    stats = summarize_trades(trades)
    assert stats["count"] >= 1
    out = Path("/tmp/nq_4h_ma200_test.html") if tmp_path is None else Path(tmp_path) / "r.html"
    path = write_html_report(out, df, trades, "NQ=F", "demo")
    text = path.read_text(encoding="utf-8")
    assert "破4小時低" in text
    assert "MA200" in text
    msg = fmt_entry(df, sigs[0])
    assert "破4h低" in msg
    assert "MA200" in msg


def main() -> int:
    test_parse_period_days()
    test_quality_from_setup()
    test_sma()
    test_detects_sep29_like()
    test_no_signal_without_ma200()
    test_simulate_and_html()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Synthetic tests for NQ 1m 高檔 M 頭跌破 MA60 做空（不連 Yahoo）。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nq.patterns import detect_m_heads  # noqa: E402
from nq_m_head import (  # noqa: E402
    ET,
    TradeResult,
    generate_signals,
    parse_period_days,
    run_backtest,
    sma,
    summarize,
    write_html_report,
)


def test_parse_period_days() -> None:
    assert parse_period_days("8d") == 8
    assert parse_period_days("30d") == 30
    assert parse_period_days("1mo") == 30
    assert parse_period_days("4w") == 28


def test_sma() -> None:
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = sma(arr, 3)
    assert np.isnan(out[1])
    assert abs(out[2] - 2.0) < 1e-9
    assert abs(out[4] - 4.0) < 1e-9


def test_summarize_short_pnl() -> None:
    class T:
        def __init__(self, pnl: float):
            self.pnl_points = pnl
            self.pnl_dollars = pnl * 20

    stats = summarize([T(40.0), T(-10.0), T(5.0)])  # type: ignore[arg-type]
    assert stats["trades"] == 3
    assert stats["wins"] == 2
    assert abs(stats["total_pnl_points"] - 35.0) < 1e-9


def _make_m_head_bars(n: int = 320) -> pd.DataFrame:
    """慢漲後末端急拉，在高檔做出夠深的雙頂，再快速跌破 MA60。"""
    close = np.zeros(n, dtype=float)
    close[0] = 20000.0
    h1, valley, h2 = 200, 218, 236

    for i in range(1, 165):
        close[i] = close[i - 1] + 0.35
    for i in range(165, h1):
        close[i] = close[i - 1] + 3.40  # 末端拉升，高峰明顯高於 MA60
    peak = close[h1 - 1] + 8.0
    close[h1] = peak
    drop = 42.0
    steps_down = valley - h1
    for i in range(h1 + 1, valley + 1):
        close[i] = peak - drop * (i - h1) / steps_down
    steps_up = h2 - valley
    for i in range(valley + 1, h2):
        close[i] = close[valley] + (peak - close[valley]) * (i - valley) / steps_up
    close[h2] = peak
    for i in range(h2 + 1, n):
        close[i] = close[i - 1] - 8.0

    high = close + 1.0
    low = close - 1.0
    peak_high = peak + 4.0
    high[h1] = peak_high
    high[h2] = peak_high
    for i in list(range(h1 - 8, h1)) + list(range(h1 + 1, h1 + 9)):
        if 0 <= i < n:
            high[i] = min(float(high[i]), peak_high - 8.0)
    for i in list(range(h2 - 8, h2)) + list(range(h2 + 1, h2 + 9)):
        if 0 <= i < n:
            high[i] = min(float(high[i]), peak_high - 8.0)
    low[valley] = close[valley] - 3.0

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


def test_detect_m_heads_geometry() -> None:
    df = _make_m_head_bars()
    patterns = detect_m_heads(df)
    assert patterns, "synthetic M-head should be detected"
    p = patterns[0]
    assert p.first_high_idx < p.second_high_idx
    assert p.second_high <= p.first_high + 0.25
    assert abs(p.first_high - p.second_high) / p.peak < 0.002
    assert p.neckline < min(p.first_high, p.second_high)


def test_detect_and_simulate_short() -> None:
    df = _make_m_head_bars()
    funnel: dict = {}
    sigs = generate_signals(df, funnel=funnel)
    assert sigs, f"expected a short after MA60 break, funnel={funnel}"
    sig = sigs[0]
    assert sig.entry < sig.stop_loss
    assert sig.target < sig.entry
    assert sig.entry < sig.ma60 + 1e-9
    assert sig.entry < sig.pattern.neckline
    assert sig.bar_idx > sig.pattern.second_high_idx

    trades = run_backtest(df, sigs, max_bars_hold=80)
    assert trades
    assert isinstance(trades[0], TradeResult)
    # 做空：價格續跌應為正損益或至少有出場
    assert trades[0].exit_idx >= trades[0].signal.bar_idx
    assert trades[0].pnl_points == trades[0].signal.entry - trades[0].exit_price


def test_reject_higher_high() -> None:
    """右峰再創高是上漲中繼，不該當成高檔 M 頭。"""
    df = _make_m_head_bars()
    high = df["high"].to_numpy(copy=True)
    h2 = 236
    high[h2] = float(high[200]) + 8.0
    df = df.copy()
    df["high"] = high
    patterns = detect_m_heads(df)
    assert all(p.second_high_idx != h2 for p in patterns)


def test_no_signal_without_ma60_break() -> None:
    df = _make_m_head_bars()
    # 把後半段改成繼續在高檔盤整，不跌破 MA60
    close = df["close"].to_numpy(copy=True)
    h2 = 236
    hold = float(close[h2])
    for i in range(h2 + 1, len(close)):
        close[i] = hold
    df = df.copy()
    df["close"] = close
    high = df["high"].to_numpy(copy=True)
    low = df["low"].to_numpy(copy=True)
    for i in range(h2 + 1, len(close)):
        high[i] = hold + 0.8
        low[i] = hold - 0.8
    df["high"] = high
    df["low"] = low
    funnel: dict = {}
    sigs = generate_signals(df, funnel=funnel)
    assert not sigs, f"should not enter without MA60 break, got {len(sigs)} funnel={funnel}"


def test_write_html_report() -> None:
    df = _make_m_head_bars()
    sigs = generate_signals(df)
    trades = run_backtest(df, sigs, max_bars_hold=80)
    out = Path("/tmp/nq_m_head_test.html")
    path = write_html_report(out, df, trades, "NQ=F", "demo")
    text = path.read_text(encoding="utf-8")
    assert "高檔M頭" in text
    if trades:
        assert "img/" in text
        img_dir = path.parent / "img"
        assert any(img_dir.glob("m01_*.png")), "expected a static trade PNG"


def main() -> int:
    test_parse_period_days()
    test_sma()
    test_summarize_short_pnl()
    test_detect_m_heads_geometry()
    test_detect_and_simulate_short()
    test_reject_higher_high()
    test_no_signal_without_ma60_break()
    test_write_html_report()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

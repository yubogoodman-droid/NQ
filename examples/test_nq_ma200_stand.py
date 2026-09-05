#!/usr/bin/env python3
"""Synthetic tests for NQ 1m 破底站上 MA200 (no Yahoo)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nq_ma200_stand import (  # noqa: E402
    ET,
    TradeResult,
    detect_signals,
    display_trades,
    in_open_skip,
    is_red_long_upper,
    overlay_15m_ma200,
    parse_period_days,
    resample_5m,
    resample_15m,
    resample_1h,
    ribbon_spread,
    ribbon_tangled,
    simulate,
    sma,
    summarize_trades,
    write_html_report,
)


def test_parse_period_days() -> None:
    assert parse_period_days("8d") == 8
    assert parse_period_days("30d") == 30
    assert parse_period_days("1mo") == 30


def test_sma() -> None:
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = sma(arr, 3)
    assert np.isnan(out[1])
    assert abs(out[2] - 2.0) < 1e-9


def test_in_open_skip() -> None:
    idx = pd.DatetimeIndex(
        [
            "2026-08-17 09:29",
            "2026-08-17 09:30",
            "2026-08-17 09:59",
            "2026-08-17 10:00",
        ],
        tz=ET,
    )
    assert in_open_skip(idx[0]) is False
    assert in_open_skip(idx[1]) is True
    assert in_open_skip(idx[2]) is True
    assert in_open_skip(idx[3]) is False


def test_red_long_upper() -> None:
    assert is_red_long_upper(100.0, 120.0, 90.0, 95.0) is True  # wick 20 >= body 5
    assert is_red_long_upper(100.0, 102.0, 90.0, 95.0) is False  # wick 2 < min 8
    assert is_red_long_upper(100.0, 120.0, 90.0, 110.0) is False  # green


def _make_setup_bars(
    n: int = 420,
    *,
    start: str = "2026-08-17 11:00",
    red_long_wick_on_entry: bool = False,
    far_from_ma200: bool = False,
    skip_under_wash: bool = False,
) -> pd.DataFrame:
    """Below MA200, dump 2h low, then stack and stand on MA200 within 1h."""
    close = np.zeros(n, dtype=float)
    close[0] = 20000.0
    # Slow grind so MA200 sits near 20020 later.
    for i in range(1, 250):
        close[i] = close[i - 1] + 0.12
    base = close[249]
    # Range just above a pinned 2h low.
    for i in range(250, 370):
        if skip_under_wash:
            close[i] = base + 8.0 + (1.0 if i % 2 == 0 else -0.5)
        else:
            close[i] = base - 25.0 + (2.0 if i % 2 == 0 else -1.0)
    break_i = 370
    close[break_i] = base - 45.0
    # Reclaim: climb back through MA200 with a bull stack.
    close[break_i + 1] = base - 10.0
    close[break_i + 2] = base + 4.0
    close[break_i + 3] = base + 12.0
    close[break_i + 4] = base + 18.0
    close[break_i + 5] = base + 22.0
    for i in range(break_i + 6, n):
        close[i] = close[i - 1] + 0.8
        if far_from_ma200:
            close[i] = close[i - 1] + 8.0

    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + 0.8
    low = np.minimum(open_, close) - 0.8
    for i in range(250, 370):
        low[i] = min(low[i], base - 8.0)
    low[break_i] = close[break_i] - 1.0
    if red_long_wick_on_entry:
        # Paint reclaim bars as red long-upper-wick so they are skipped.
        for j in range(break_i + 2, break_i + 12):
            open_[j] = close[j] + 6.0
            high[j] = open_[j] + 16.0
            low[j] = close[j] - 1.0

    idx = pd.date_range(start, periods=n, freq="1min", tz=ET)
    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": np.full(n, 80.0),
        },
        index=idx,
    )


def test_detect_happy_path() -> None:
    df = _make_setup_bars()
    sigs = detect_signals(df, min_5m_ribbon=0.0)
    assert sigs, "expected a stand-on-MA200 signal after the 2h dump"
    sig = sigs[0]
    assert sig.entry_idx > sig.break_idx
    assert sig.entry_idx - sig.break_idx <= 60
    assert sig.ma5 > sig.ma10 > sig.ma20 > sig.ma30 > sig.ma60
    assert 0 < sig.dist_ma200 <= 30
    assert sig.under_streak >= 15
    assert abs(sig.stop_price - (sig.ma200 - 10)) < 1e-6
    assert abs(sig.target_price - (sig.entry_price + 100)) < 1e-6


def test_skip_red_long_wick() -> None:
    df = _make_setup_bars(red_long_wick_on_entry=True)
    sigs = detect_signals(df, min_5m_ribbon=0.0)
    assert not sigs, "red long-upper-wick reclaim bars should be skipped"


def test_skip_open_hour() -> None:
    df = _make_setup_bars(start="2026-08-17 08:20")
    # Break around 08:20+370min = 14:30? 8:20 + 370 min = 14:30. Need 9:30 window.
    # Restart so the reclaim lands in 9:30–10:00.
    df = _make_setup_bars(start="2026-08-17 03:00")
    # 03:00 + 370 min = 09:10 break; reclaim ~09:13–09:20 still before 9:30.
    # Shift later: 03:20 + 370 = 09:30.
    df = _make_setup_bars(start="2026-08-17 03:22")
    sigs = detect_signals(df, min_5m_ribbon=0.0)
    for sig in sigs:
        assert not in_open_skip(df.index[sig.entry_idx])


def test_skip_no_under_wash() -> None:
    df = _make_setup_bars(skip_under_wash=True)
    sigs = detect_signals(df, min_5m_ribbon=0.0)
    assert not sigs, "without 15 bars under MA200 there should be no entry"


def test_simulate_target_and_stop() -> None:
    df = _make_setup_bars()
    sigs = detect_signals(df, min_5m_ribbon=0.0)
    assert sigs
    trades = simulate(df, sigs)
    assert trades
    assert isinstance(trades[0], TradeResult)
    assert trades[0].exit_idx >= trades[0].entry_idx
    # Push through +100 after entry.
    df2 = df.copy()
    j = sigs[0].entry_idx
    df2.iloc[j + 3, df2.columns.get_loc("High")] = sigs[0].entry_price + 120
    df2.iloc[j + 3, df2.columns.get_loc("Close")] = sigs[0].entry_price + 110
    trades2 = simulate(df2, sigs)
    assert trades2[0].exit_reason == "target"
    assert abs(trades2[0].pnl_points - 100) < 1e-6

    df3 = df.copy()
    df3.iloc[j + 2, df3.columns.get_loc("Low")] = sigs[0].stop_price - 1
    trades3 = simulate(df3, sigs)
    assert trades3[0].exit_reason == "stop"


def test_breakeven_after_plus_60() -> None:
    df = _make_setup_bars()
    sigs = detect_signals(df, min_5m_ribbon=0.0)
    assert sigs
    j = sigs[0].entry_idx
    entry = sigs[0].entry_price
    df2 = df.copy()
    df2.iloc[j + 2, df2.columns.get_loc("High")] = entry + 65
    df2.iloc[j + 4, df2.columns.get_loc("Low")] = entry - 1
    df2.iloc[j + 4, df2.columns.get_loc("Close")] = entry - 1
    trades = simulate(df2, sigs, breakeven_after=60.0)
    assert trades[0].exit_reason == "trail"
    assert abs(trades[0].exit_price - entry) < 1e-6


def test_write_html(tmp_path: Path | None = None) -> None:
    df = _make_setup_bars()
    sigs = detect_signals(df, min_5m_ribbon=0.0)
    trades = simulate(df, sigs)
    out = Path("/tmp/nq_ma200_stand_test.html") if tmp_path is None else Path(tmp_path) / "r.html"
    path = write_html_report(out, df, trades, "NQ=F", "demo")
    text = path.read_text(encoding="utf-8")
    assert "破底站上 MA200" in text
    if trades:
        assert "<img src='img/" in text
        assert "5m 對照" in text
        assert "15m 對照" in text
        assert "1h 對照" in text
        assert "距200日" not in text
        assert "距15mMA200" in text
        assert "進場距 15m MA200" in text
        assert "停損 MA200−10" in text
        assert "5m連2根收在破底下" not in text
        assert any((path.parent / "img").glob("t01_*.png"))
        assert any((path.parent / "img").glob("t01_*_5m.png"))
        assert any((path.parent / "img").glob("t01_*_15m.png"))
        assert any((path.parent / "img").glob("t01_*_1h.png"))
        if any(t.pnl_points > 0 for t in trades) and any(t.pnl_points <= 0 for t in trades):
            assert text.find("賺錢") < text.find("賠錢")


def test_display_trades_wins_first() -> None:
    class T:
        def __init__(self, pnl: float, entry_idx: int):
            self.pnl_points = pnl
            self.entry_idx = entry_idx

    ordered = display_trades([T(-10, 1), T(100, 5), T(-20, 3), T(100, 2)])  # type: ignore[list-item]
    assert [t.entry_idx for t in ordered] == [2, 5, 1, 3]


def test_resample_5m() -> None:
    idx = pd.date_range("2026-08-17 11:00", periods=90, freq="1min", tz=ET)
    close = np.arange(90, dtype=float) + 100.0
    df = pd.DataFrame(
        {
            "Open": close,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Volume": np.ones(90),
        },
        index=idx,
    )
    m5 = resample_5m(df)
    m15 = resample_15m(df)
    m1h = resample_1h(df)
    assert len(m5) >= 1
    assert len(m15) >= 1
    assert len(m1h) >= 1
    assert {"Open", "High", "Low", "Close"}.issubset(m5.columns)
    assert {"Open", "High", "Low", "Close"}.issubset(m15.columns)
    assert {"Open", "High", "Low", "Close"}.issubset(m1h.columns)


def test_overlay_15m_ma200() -> None:
    n = 200 * 15 + 45
    idx = pd.date_range("2026-01-05 00:00", periods=n, freq="1min", tz=ET)
    close = np.full(n, 100.0)
    close[-15:] = 110.0
    df = pd.DataFrame(
        {"Open": close, "High": close + 1, "Low": close - 1, "Close": close, "Volume": np.ones(n)},
        index=idx,
    )
    out = overlay_15m_ma200(df)
    assert np.isnan(out[100])
    assert not np.isnan(out[-1])
    assert abs(out[-1] - 100.0) < 0.6


def test_overlay_15m_ma200_uses_long_history() -> None:
    idx_1m = pd.date_range("2026-03-01 10:00", periods=80, freq="1min", tz=ET)
    close_1m = np.full(80, 110.0)
    df_1m = pd.DataFrame(
        {"Open": close_1m, "High": close_1m + 1, "Low": close_1m - 1, "Close": close_1m},
        index=idx_1m,
    )
    idx_15 = pd.date_range("2026-02-01 00:00", periods=220, freq="15min", tz=ET)
    close_15 = np.full(220, 100.0)
    df_15 = pd.DataFrame(
        {"Open": close_15, "High": close_15 + 1, "Low": close_15 - 1, "Close": close_15},
        index=idx_15,
    )
    out = overlay_15m_ma200(df_1m, df_15)
    assert not np.isnan(out[-1])
    assert abs(out[-1] - 100.0) < 1e-6
    assert np.isnan(overlay_15m_ma200(df_1m)[-1])


def test_ribbon_helpers() -> None:
    assert abs(ribbon_spread(100.0, 110.0, 105.0, 108.0, 102.0) - 10.0) < 1e-9
    assert ribbon_tangled(100.0, 110.0, 105.0, 108.0, min_spread=17.0) is True
    assert ribbon_tangled(100.0, 140.0, 120.0, 130.0, min_spread=17.0) is False


def test_default_has_no_5m_tangle_filter() -> None:
    df = _make_setup_bars()
    assert detect_signals(df), "default is the original 7 rules; 5m ribbon is display-only"


def test_summarize() -> None:
    class T:
        def __init__(self, pnl: float):
            self.pnl_points = pnl

    stats = summarize_trades([T(100.0), T(-20.0)])  # type: ignore[list-item]
    assert stats["count"] == 2
    assert stats["wins"] == 1
    assert abs(stats["total_points"] - 80.0) < 1e-9


def main() -> int:
    test_parse_period_days()
    test_sma()
    test_in_open_skip()
    test_red_long_upper()
    test_detect_happy_path()
    test_skip_red_long_wick()
    test_skip_open_hour()
    test_skip_no_under_wash()
    test_simulate_target_and_stop()
    test_breakeven_after_plus_60()
    test_write_html()
    test_display_trades_wins_first()
    test_resample_5m()
    test_overlay_15m_ma200()
    test_overlay_15m_ma200_uses_long_history()
    test_ribbon_helpers()
    test_default_has_no_5m_tangle_filter()
    test_summarize()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

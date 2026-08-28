#!/usr/bin/env python3
"""Synthetic tests for 1h MA25 undercut + reclaim (no Binance)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from binance_1h_ma25_reclaim import (  # noqa: E402
    TradeResult,
    classify_shape,
    detect_signals,
    quality_of,
    simulate,
    sma,
    summarize_trades,
    write_html_report,
)


def test_sma() -> None:
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = sma(arr, 3)
    assert np.isnan(out[1])
    assert abs(out[2] - 2.0) < 1e-9
    assert abs(out[4] - 4.0) < 1e-9


def test_quality_of() -> None:
    assert quality_of(0.03, 1.5, "W", True)[1] == "A"
    assert quality_of(0.03, 1.0, "V", False)[1] == "C"
    assert quality_of(0.03, 1.5, "V", False)[1] == "B"


def test_classify_v_and_w() -> None:
    n = 20
    low = np.full(n, 10.0)
    high = np.full(n, 10.4)
    low[8] = 9.4
    assert classify_shape(low, high, 4, 16) == "V"

    low2 = np.full(n, 10.0)
    high2 = np.full(n, 10.5)
    low2[6] = 9.50
    high2[10] = 10.80
    low2[14] = 9.52
    assert classify_shape(low2, high2, 4, 17) == "W"


def _base_index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2026-08-01 00:00", periods=n, freq="1h", tz="Asia/Taipei")


def _make_reclaim_bars(n: int = 80, *, depth: float = 0.04, below: int = 14) -> pd.DataFrame:
    """Ride above MA25, dump below it, print a bottom, then close back above."""
    close = np.zeros(n, dtype=float)
    close[0] = 100.0
    for i in range(1, 40):
        close[i] = close[i - 1] + 0.12
    dump = 40
    floor = close[dump - 1] * (1.0 - depth)
    for i in range(dump, dump + below):
        # grind down to the floor then lift toward MA
        t = (i - dump) / max(below - 1, 1)
        close[i] = close[dump - 1] + (floor - close[dump - 1]) * min(1.0, t * 1.4 if t < 0.55 else (1.4 - t))
    close[dump + below - 1] = close[dump - 1] * 0.985
    close[dump + below] = close[dump - 1] + 0.80
    for i in range(dump + below + 1, n):
        close[i] = close[i - 1] + 0.25
    high = close + 0.35
    low = close - 0.35
    low[dump + 4] = floor
    high[dump + 4] = close[dump + 4] + 0.15
    return pd.DataFrame(
        {
            "Open": np.r_[close[0], close[:-1]],
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": np.r_[np.full(dump + below, 80.0), [220.0], np.full(n - dump - below - 1, 90.0)],
        },
        index=_base_index(n),
    )


def test_detect_reclaim() -> None:
    df = _make_reclaim_bars()
    funnel: dict = {}
    sigs = detect_signals(df, funnel=funnel)
    assert sigs, f"expected a reclaim, funnel={funnel}"
    sig = sigs[0]
    assert sig.entry_idx > sig.break_idx
    assert sig.entry_price > sig.stop_price
    assert sig.bars_below >= 4
    assert sig.depth_pct >= 0.018
    assert sig.quality in {"A", "B", "C"}
    assert df["Close"].iloc[sig.entry_idx] > sig.ma25


def test_shallow_rejected() -> None:
    df = _make_reclaim_bars(depth=0.006, below=8)
    sigs = detect_signals(df, min_depth_pct=0.028)
    assert not sigs, "0.6% poke should not count as 破底"


def test_one_bar_pop_keeps_episode() -> None:
    df = _make_reclaim_bars(depth=0.045, below=16)
    close = df["Close"].to_numpy(float)
    # W 中間兩根假站上，然後再破底
    df.iloc[44, df.columns.get_loc("Close")] = close[39] + 0.4
    df.iloc[44, df.columns.get_loc("High")] = close[39] + 0.6
    df.iloc[45, df.columns.get_loc("Close")] = close[39] + 0.25
    df.iloc[45, df.columns.get_loc("High")] = close[39] + 0.45
    sigs = detect_signals(df)
    assert sigs, "a 1–2 bar bounce above MA25 should not split the dip"
    assert sigs[0].bars_below >= 8


def test_still_below_no_signal() -> None:
    df = _make_reclaim_bars()
    # chop the reclaim bar off
    cut = df.iloc[:50].copy()
    # force the tail to stay under a rising MA
    cut.loc[cut.index[-6]:, "Close"] = 90.0
    cut.loc[cut.index[-6]:, "Low"] = 89.5
    cut.loc[cut.index[-6]:, "High"] = 90.4
    sigs = detect_signals(cut)
    assert not sigs


def test_simulate_target() -> None:
    df = _make_reclaim_bars()
    sigs = detect_signals(df)
    trades = simulate(df, sigs, max_hold=40)
    assert trades
    assert isinstance(trades[0], TradeResult)
    assert trades[0].exit_idx >= trades[0].entry_idx
    assert trades[0].exit_reason in {"target", "stop", "lost_ma25", "timeout"}


def test_summarize() -> None:
    class T:
        def __init__(self, pnl: float, quality: str = "A"):
            self.pnl_pct = pnl
            self.quality = quality

    stats = summarize_trades([T(2.0, "A"), T(-1.0, "B")])  # type: ignore[list-item]
    assert stats["count"] == 2
    assert stats["wins"] == 1
    assert abs(stats["total_pct"] - 1.0) < 1e-9


def test_select_card_hits() -> None:
    from binance_1h_ma25_reclaim import Hit, select_card_hits  # noqa: WPS433

    df = _make_reclaim_bars()
    sigs = detect_signals(df)
    trades = simulate(df, sigs, max_hold=40)
    assert trades
    hits = [Hit("FOOUSDT", df, trades[0]), Hit("AVGOUSDT", df, trades[0])]
    picked = select_card_hits(hits, recent_hours=0, keep_symbols=("AVGOUSDT",))
    assert [h.symbol for h in picked] == ["AVGOUSDT"]


def test_write_html(tmp_path: Path | None = None) -> None:
    df = _make_reclaim_bars()
    sigs = detect_signals(df)
    trades = simulate(df, sigs, max_hold=40)
    from binance_1h_ma25_reclaim import Hit  # noqa: WPS433

    out = Path("/tmp/ma25_reclaim_test.html") if tmp_path is None else Path(tmp_path) / "r.html"
    hits = [Hit("AVGOUSDT", df, t) for t in trades]
    path = write_html_report(out, hits, days=7, scanned=1)
    text = path.read_text(encoding="utf-8")
    assert "MA25" in text
    assert "下破底" in text
    if trades:
        assert "<img src='img/" in text
        assert any((path.parent / "img").glob("t01_*.png"))


def main() -> int:
    test_sma()
    test_quality_of()
    test_classify_v_and_w()
    test_detect_reclaim()
    test_shallow_rejected()
    test_still_below_no_signal()
    test_simulate_target()
    test_summarize()
    test_select_card_hits()
    test_write_html()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

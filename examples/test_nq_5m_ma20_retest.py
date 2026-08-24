#!/usr/bin/env python3
"""Synthetic tests for NQ 5m 破翻回踩 MA20 (no Yahoo)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nq.ma20_retest import (  # noqa: E402
    TradeResult,
    detect_signals,
    quality_at_entry,
    simulate,
    sma,
    summarize_trades,
)
from nq_5m_ma20_retest import ET, parse_period_days, write_html_report  # noqa: E402


def test_parse_period_days() -> None:
    assert parse_period_days("8d") == 8
    assert parse_period_days("30d") == 30
    assert parse_period_days("1mo") == 30
    assert parse_period_days("4w") == 28


def test_quality_at_entry() -> None:
    assert quality_at_entry(20.0, 19.0, 18.0, 2.0)[1] == "A"
    assert quality_at_entry(20.0, 21.0, 18.0, -1.0)[1] == "B"
    assert quality_at_entry(10.0, 11.0, 12.0, -1.0)[1] == "C"


def test_sma() -> None:
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = sma(arr, 3)
    assert np.isnan(out[1])
    assert abs(out[2] - 2.0) < 1e-9


def test_summarize_trades() -> None:
    class T:
        def __init__(self, pnl: float, quality: str = "A"):
            self.pnl_points = pnl
            self.quality = quality

    stats = summarize_trades([T(10.0, "A"), T(-4.0, "B"), T(2.0, "A")])
    assert stats["count"] == 3
    assert stats["wins"] == 2
    assert abs(stats["total_points"] - 8.0) < 1e-9


def _range_then_dump_reclaim_retest(n: int = 220) -> pd.DataFrame:
    """2h 盤整 → 破底 → 收復 MA20 → 離開 → 回踩（對齊 08-24 藍圈節奏）。

    06:30 起算：bar 90≈14:00 破底，bar 108≈15:30 回踩，落在 RTH。
    """
    close = np.full(n, 29200.0)
    high = np.full(n, 29204.0)
    low = np.full(n, 29196.0)
    for i in range(n):
        close[i] = 29200.0 + (3.0 if i % 2 == 0 else -2.0)
        high[i] = close[i] + 4.0
        low[i] = max(close[i] - 4.0, 29190.0)

    # dump → trough 28946.75
    path = [
        (90, 29120.0, 18.0, 8.0),
        (91, 29070.0, 18.0, 8.0),
        (92, 28980.0, 18.0, 8.0),
        (93, 28955.0, 18.0, 8.0),
        (94, 28950.0, 18.0, 3.25),
        (95, 28980.0, 18.0, 8.0),
        (96, 28990.0, 18.0, 8.0),
        (97, 28978.0, 18.0, 8.0),
        (98, 29012.0, 12.0, 14.0),
        (99, 29076.0, 12.0, 14.0),
        (100, 29081.0, 12.0, 14.0),
        (101, 29085.0, 12.0, 14.0),
        (102, 29106.0, 12.0, 14.0),
        (103, 29120.0, 10.0, 8.0),
        (104, 29128.0, 10.0, 8.0),
        (105, 29122.0, 10.0, 8.0),
        (106, 29110.0, 8.0, 10.0),
        (107, 29090.0, 8.0, 12.0),
        (108, 29070.0, 15.0, 8.0),  # 回踩：低點貼 MA20，不刺穿太多
    ]
    for i, px, up, dn in path:
        close[i] = px
        high[i] = px + up
        low[i] = px - dn
    low[94] = 28946.75

    for i in range(109, n):
        close[i] = close[i - 1] + 6.0
        high[i] = close[i] + 5.0
        low[i] = close[i] - 5.0

    idx = pd.date_range("2026-08-24 06:30", periods=n, freq="5min", tz=ET)
    return pd.DataFrame(
        {
            "Open": np.r_[close[0], close[:-1]],
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": np.full(n, 200.0),
        },
        index=idx,
    )


def test_detects_retest_not_reclaim() -> None:
    df = _range_then_dump_reclaim_retest()
    sigs = detect_signals(df, session="day")
    assert sigs, "expected a 回踩 MA20 signal on the synthetic dump"
    sig = sigs[0]
    assert sig.entry_idx > sig.reclaim_idx
    assert sig.reclaim_idx > sig.trough_idx
    assert sig.entry_idx - sig.reclaim_idx >= 3
    assert sig.entry_price > sig.stop_price
    # 進場應靠近 MA20，而不是收復當根遠拋
    assert abs(sig.entry_price - sig.ma20) < 80.0
    assert df["Low"].iloc[sig.entry_idx] <= sig.ma20 + 8.0 + 1e-6


def test_no_entry_if_never_leaves_ma20() -> None:
    df = _range_then_dump_reclaim_retest()
    # hug MA20 after reclaim: never put 3 bars with low > MA20+10
    sigs = detect_signals(df, session="day", leave_bars=3, leave_buffer=80.0)
    assert not sigs


def test_simulate_exits() -> None:
    df = _range_then_dump_reclaim_retest()
    sigs = detect_signals(df, session="day")
    trades = simulate(df, sigs)
    assert trades
    assert isinstance(trades[0], TradeResult)
    assert trades[0].exit_idx >= trades[0].entry_idx


def test_write_html_report(tmp_path: Path | None = None) -> None:
    df = _range_then_dump_reclaim_retest()
    sigs = detect_signals(df, session="day")
    trades = simulate(df, sigs)
    out = Path("/tmp/nq_5m_ma20_test.html") if tmp_path is None else Path(tmp_path) / "r.html"
    path = write_html_report(out, df, trades, "NQ=F", "demo")
    text = path.read_text(encoding="utf-8")
    assert "回踩 MA20" in text
    if trades:
        assert "<img src='img/" in text
        img_dir = path.parent / "img"
        assert any(img_dir.glob("t01_*.png")), "expected a static trade PNG"


def main() -> int:
    test_parse_period_days()
    test_quality_at_entry()
    test_sma()
    test_summarize_trades()
    test_detects_retest_not_reclaim()
    test_no_entry_if_never_leaves_ma20()
    test_simulate_exits()
    test_write_html_report()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

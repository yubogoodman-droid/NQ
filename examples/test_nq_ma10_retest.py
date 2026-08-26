#!/usr/bin/env python3
"""Synthetic tests for NQ 5m 破底 → 站上 MA10 → 回踩 MA10 (no Yahoo)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nq.ma10_retest import (  # noqa: E402
    TradeResult,
    detect_signals,
    quality_from_setup,
    simulate,
    sma,
    summarize_trades,
)
from nq_ma10_retest import ET, make_demo_bars, write_html_report  # noqa: E402


def test_quality_from_setup() -> None:
    assert quality_from_setup(50.0, 4.0, True) == (3, "A")
    assert quality_from_setup(50.0, 20.0, False) == (1, "B")
    assert quality_from_setup(10.0, 20.0, False) == (0, "C")


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
    assert stats["by_quality"]["A"]["n"] == 2


def _make_setup(
    *,
    pullback: bool = True,
    lose_stand: bool = False,
    runaway: bool = False,
    n: int = 120,
) -> pd.DataFrame:
    close = np.full(n, 20000.0)
    high = close + 2.0
    low = close - 2.0
    open_ = close.copy()
    dump = 70
    close[dump] = 19940.0
    open_[dump] = 19990.0
    high[dump] = 19995.0
    low[dump] = 19930.0
    close[dump + 1] = 19950.0
    open_[dump + 1] = 19942.0
    high[dump + 1] = 19955.0
    low[dump + 1] = 19928.0
    close[dump + 2] = 19980.0
    open_[dump + 2] = 19952.0
    high[dump + 2] = 19988.0
    low[dump + 2] = 19950.0
    close[dump + 3] = 20040.0
    open_[dump + 3] = 19985.0
    high[dump + 3] = 20048.0
    low[dump + 3] = 19982.0
    close[dump + 4] = 20055.0
    open_[dump + 4] = 20038.0
    high[dump + 4] = 20062.0
    low[dump + 4] = 20036.0
    close[dump + 5] = 20070.0
    open_[dump + 5] = 20054.0
    high[dump + 5] = 20078.0
    low[dump + 5] = 20050.0
    close[dump + 6] = 20078.0
    open_[dump + 6] = 20068.0
    high[dump + 6] = 20085.0
    low[dump + 6] = 20062.0
    if lose_stand:
        close[dump + 7] = 19970.0
        open_[dump + 7] = 20070.0
        high[dump + 7] = 20072.0
        low[dump + 7] = 19960.0
        for i in range(dump + 8, n):
            close[i] = 19965.0
            open_[i] = 19970.0
            high[i] = 19975.0
            low[i] = 19955.0
    elif runaway:
        for i in range(dump + 7, n):
            close[i] = close[i - 1] + 12.0
            open_[i] = close[i - 1]
            high[i] = close[i] + 2.0
            low[i] = open_[i] + 4.0
    elif pullback:
        close[dump + 7] = 20020.0
        open_[dump + 7] = 20072.0
        high[dump + 7] = 20076.0
        low[dump + 7] = 19995.0
        for i in range(dump + 8, n):
            close[i] = close[i - 1] + 8.0
            open_[i] = close[i - 1]
            high[i] = close[i] + 2.0
            low[i] = open_[i] - 1.0
    else:
        for i in range(dump + 7, n):
            close[i] = 20070.0
            open_[i] = 20068.0
            high[i] = 20080.0
            low[i] = 20050.0

    idx = pd.date_range("2026-07-28 08:00", periods=n, freq="5min", tz=ET)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": np.full(n, 80.0)},
        index=idx,
    )


def test_detect_break_reclaim_retest() -> None:
    df = _make_setup(pullback=True)
    sigs = detect_signals(df)
    assert sigs, "expected a 回踩 MA10 long after 破底 and 站上"
    sig = sigs[0]
    assert sig.entry_idx > sig.reclaim_idx > sig.break_idx
    assert sig.entry_price > sig.stop_price
    assert sig.pierce >= 0
    close = df["Close"].to_numpy(float)
    ma10 = sma(close, 10)
    assert close[sig.reclaim_idx] > ma10[sig.reclaim_idx]
    assert df["Low"].iloc[sig.entry_idx] <= ma10[sig.entry_idx]
    assert close[sig.entry_idx] >= ma10[sig.entry_idx]


def test_retry_after_brief_lost_stand() -> None:
    df = _make_setup(pullback=True)
    close = df["Close"].to_numpy(float).copy()
    low = df["Low"].to_numpy(float).copy()
    high = df["High"].to_numpy(float).copy()
    open_ = df["Open"].to_numpy(float).copy()
    dump = 70
    ma10 = sma(close, 10)
    # 站上後下一根收盤只比 MA10 低 2 點，之後再站上並回踩
    close[dump + 4] = float(ma10[dump + 4]) - 2.0
    open_[dump + 4] = float(ma10[dump + 4]) + 8.0
    high[dump + 4] = open_[dump + 4] + 1.0
    low[dump + 4] = close[dump + 4] - 2.0
    df = df.copy()
    df["Close"] = close
    df["Open"] = open_
    df["High"] = high
    df["Low"] = low
    sigs = detect_signals(df)
    assert sigs, "a 2-point dip through MA10 should not kill the 破底; wait to 站上 again"


def test_no_entry_when_stand_is_lost() -> None:
    df = _make_setup(lose_stand=True)
    sigs = detect_signals(df)
    assert not sigs, "closing back below MA10 should cancel the 回踩 setup"


def test_no_entry_when_price_runs_away() -> None:
    df = _make_setup(runaway=True)
    sigs = detect_signals(df)
    assert not sigs, "no 回踩 if price never touches MA10 again"


def test_simulate_hits_target() -> None:
    df = _make_setup(pullback=True)
    sigs = detect_signals(df)
    trades = simulate(df, sigs, max_hold=24)
    assert trades
    assert isinstance(trades[0], TradeResult)
    assert trades[0].exit_idx >= trades[0].entry_idx
    assert trades[0].exit_reason in {"target", "stop", "time"}
    assert trades[0].pnl_points > 0


def test_default_lookback_is_four_hours() -> None:
    import inspect

    default = inspect.signature(detect_signals).parameters["lookback_bars"].default
    assert default == 48, "5m × 48 = 4 hours"


def test_demo_bars_and_html(tmp_path: Path | None = None) -> None:
    df = make_demo_bars()
    sigs = detect_signals(df)
    assert sigs
    trades = simulate(df, sigs)
    out = Path("/tmp/nq_ma10_retest_test.html") if tmp_path is None else Path(tmp_path) / "r.html"
    path = write_html_report(out, df, trades, "NQ=F", "demo")
    text = path.read_text(encoding="utf-8")
    assert "破底回踩 MA10" in text
    assert "<img src='img/" in text
    img_dir = path.parent / "img"
    assert any(img_dir.glob("t01_*.png")), "expected a static trade PNG"


def main() -> int:
    test_quality_from_setup()
    test_sma()
    test_summarize_trades()
    test_detect_break_reclaim_retest()
    test_retry_after_brief_lost_stand()
    test_no_entry_when_stand_is_lost()
    test_no_entry_when_price_runs_away()
    test_simulate_hits_target()
    test_default_lookback_is_four_hours()
    test_demo_bars_and_html()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

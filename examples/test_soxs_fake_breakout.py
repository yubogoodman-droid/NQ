#!/usr/bin/env python3
"""SOXS 假突破做空的合成測試。不打幣安。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from soxs_fake_breakout import (  # noqa: E402
    Params,
    in_session,
    run_backtest,
    score_window,
    simulate_exit,
)


def _plant_fakeout(n: int = 80, start: str = "2026-11-12 21:00", peak: float = 103.35) -> pd.DataFrame:
    """均線墊高後刺破前高，下一根收弱，再回測均線失敗。"""
    idx = pd.date_range(start, periods=n, freq="min", tz="Asia/Taipei")
    c = np.full(n, 100.0)
    c[10:36] = 102.2
    c[36:40] = 100.4
    c[40] = 101.6
    c[41] = 100.2
    c[42] = 100.7
    c[43] = 97.0
    c[44] = 98.0
    h = np.maximum(c + 0.15, c)
    l = c - 0.15
    o = np.r_[c[0], c[:-1]]
    h[28] = 103.0
    h[29:40] = np.minimum(h[29:40], 102.4)
    h[40] = peak
    l[40] = 101.2
    o[40] = 101.0
    h[41] = 101.2
    l[41] = 100.0
    o[41] = 101.4
    o[42] = 100.4
    h[42] = 102.3
    l[42] = 100.2
    o[43] = 100.6
    h[43] = 100.8
    l[43] = 96.0
    return pd.DataFrame(
        {"Open": o, "High": h, "Low": l, "Close": c, "Volume": np.full(n, 1000.0)},
        index=idx,
    )


def _params(**overrides) -> Params:
    base = dict(
        lookback=20,
        min_sep=5,
        break_window=10,
        retest_window=6,
        max_hold=8,
        days=0,
        session="all",
        mode="limit",
    )
    base.update(overrides)
    return Params(**base)


def test_session_windows() -> None:
    hit = pd.Timestamp("2026-11-12 22:26", tz="Asia/Taipei")
    midday = pd.Timestamp("2026-11-12 12:00", tz="Asia/Taipei")
    early = pd.Timestamp("2026-11-12 18:00", tz="Asia/Taipei")
    friday_rth = pd.Timestamp("2026-11-13 04:30", tz="Asia/Taipei")
    saturday = pd.Timestamp("2026-11-14 22:00", tz="Asia/Taipei")
    assert in_session(hit, "tpe_us")
    assert in_session(hit, "ny")
    assert not in_session(midday, "tpe_us")
    assert not in_session(midday, "ny")
    assert not in_session(early, "tpe_us")
    assert not in_session(early, "ny")
    assert in_session(friday_rth, "tpe_us")
    assert in_session(friday_rth, "ny")
    assert not in_session(saturday, "tpe_us")
    assert not in_session(saturday, "ny")
    assert in_session(saturday, "all")


def test_limit_retest_hits_target_without_future_bars() -> None:
    df = _plant_fakeout()
    trades, funnel = run_backtest(df, _params())
    assert funnel["entered"] == 1
    assert len(trades) == 1
    t = trades[0]
    assert t.entry_idx == 42
    assert t.peak_high == 103.35
    assert t.exit_reason == "target"
    assert abs(t.r_multiple - 2.5) < 1e-9
    assert t.net_pct > 0

    cut = df.iloc[:43].copy()
    early, _ = run_backtest(cut, _params())
    assert len(early) == 1
    assert early[0].entry_idx == 42
    assert early[0].entry_price == t.entry_price
    assert early[0].exit_reason == "open"

    tainted = df.copy()
    tainted.iloc[55, tainted.columns.get_loc("Low")] = 1.0
    tainted.iloc[55, tainted.columns.get_loc("High")] = 200.0
    again, _ = run_backtest(tainted, _params())
    assert again[0].entry_price == t.entry_price
    assert again[0].exit_idx == t.exit_idx
    assert again[0].exit_price == t.exit_price


def test_same_bar_stop_beats_target() -> None:
    high = np.array([101.0, 110.0, 90.0])
    low = np.array([99.0, 90.0, 90.0])
    close = np.array([100.0, 95.0, 95.0])
    idx, px, reason = simulate_exit(high, low, close, 1, 100.0, 105.0, 90.0, max_hold=2)
    assert (idx, px, reason) == (1, 105.0, "stop")


def test_runaway_overshoot_is_skipped() -> None:
    df = _plant_fakeout(peak=120.0)
    trades, funnel = run_backtest(df, _params())
    assert trades == []
    assert funnel["entered"] == 0
    assert funnel["runaway"] >= 1


def test_half_hour_score_ignores_later_target() -> None:
    df = _plant_fakeout()
    # 把 7 分鐘後的停利移到第 40 根之後，半小時結算不該算贏在停利。
    df.iloc[43, df.columns.get_loc("Low")] = 100.4
    df.iloc[43, df.columns.get_loc("Close")] = 100.5
    df.iloc[75, df.columns.get_loc("Low")] = 90.0
    df.iloc[75, df.columns.get_loc("Close")] = 91.0
    trades, _ = run_backtest(df, _params())
    assert len(trades) == 1
    early = score_window(df, trades, 30, fee_pct=0.0006)
    assert early["count"] == 1
    assert early["reasons"].get("target", 0) == 0
    later = score_window(df, trades, 60, fee_pct=0.0006)
    assert later["reasons"].get("target", 0) == 1


def test_outside_session_is_ignored() -> None:
    df = _plant_fakeout(start="2026-11-14 21:00")
    trades, funnel = run_backtest(df, _params(session="tpe_us"))
    assert trades == []
    assert funnel["entered"] == 0


if __name__ == "__main__":
    test_session_windows()
    test_limit_retest_hits_target_without_future_bars()
    test_same_bar_stop_beats_target()
    test_runaway_overshoot_is_skipped()
    test_outside_session_is_ignored()
    print("ok")

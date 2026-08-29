#!/usr/bin/env python3
"""Synthetic tests for 15m 同時跌破 99/120/200 + 7/14/25 空頭排列做空。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from binance_15m_ma_break_short import (  # noqa: E402
    TZ,
    add_mas,
    detect_signals,
    draw_trade_png,
    hourly_snapshot,
    resample_1h,
    simulate,
    summarize,
)


def _bars(close: np.ndarray) -> pd.DataFrame:
    n = len(close)
    high = close + 0.04
    low = close - 0.04
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    idx = pd.date_range("2026-08-20 00:00", periods=n, freq="15min", tz=TZ)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": np.full(n, 1000.0)},
        index=idx,
    )


def _breakdown_series(n: int = 280, crash: float = 0.90, after: str = "down") -> pd.DataFrame:
    """長均黏在價下，短均先轉空，再一根大陰線從三條之上打穿到三條之下。"""
    close = np.zeros(n, dtype=float)
    close[0] = 20.0
    for i in range(1, 210):
        close[i] = 20.0 + 0.008 * i + 0.03 * np.sin(i / 7.0)
    # 淺回讓 7<14<25，但價仍在三條長均之上
    for i in range(210, 230):
        close[i] = close[i - 1] - 0.006
    break_i = 230
    close[break_i] = close[break_i - 1] - crash
    if after == "down":
        for i in range(break_i + 1, n):
            close[i] = close[i - 1] - 0.12
    else:
        close[break_i + 1] = close[break_i - 1] + 0.35
        for i in range(break_i + 2, n):
            close[i] = close[i - 1] + 0.04
        df = _bars(close)
        df.iloc[break_i, df.columns.get_loc("high")] = close[break_i - 1] + 0.08
        df.iloc[break_i, df.columns.get_loc("open")] = close[break_i - 1]
        df.iloc[break_i, df.columns.get_loc("low")] = close[break_i] - 0.06
        df.iloc[break_i + 1, df.columns.get_loc("high")] = close[break_i + 1] + 0.06
        df.iloc[break_i + 1, df.columns.get_loc("low")] = close[break_i] - 0.02
        return df
    df = _bars(close)
    df.iloc[break_i, df.columns.get_loc("high")] = close[break_i - 1] + 0.08
    df.iloc[break_i, df.columns.get_loc("open")] = close[break_i - 1]
    df.iloc[break_i, df.columns.get_loc("low")] = close[break_i] - 0.06
    return df


def test_detects_simultaneous_break_and_short_stack() -> None:
    df = add_mas(_breakdown_series(after="down"))
    sigs = detect_signals(df, "TESTUSDT")
    assert len(sigs) >= 1
    s = sigs[0]
    assert s.ma7 < s.ma14 < s.ma25
    assert s.entry < s.ma99 and s.entry < s.ma120 and s.entry < s.ma200
    assert s.entry < s.stop
    assert s.target < s.entry


def test_no_signal_without_short_stack() -> None:
    df = _breakdown_series(crash=0.35, after="down")
    # 幾乎沒有回落，短均仍多頭，只輕輕跌一下，不該進場
    close = df["close"].to_numpy(copy=True)
    close[:230] = 20.0 + np.linspace(0, 2.4, 230)
    close[230] = close[229] - 0.15
    close[231:] = close[230]
    df = add_mas(_bars(close))
    df.iloc[230, df.columns.get_loc("high")] = close[229] + 0.02
    sigs = detect_signals(df, "TESTUSDT")
    # 要嘛沒打穿三條，要嘛短均不是 7<14<25
    for s in sigs:
        assert s.ma7 < s.ma14 < s.ma25


def test_short_take_profit() -> None:
    df = add_mas(_breakdown_series(after="down"))
    trades = simulate(df, detect_signals(df, "TESTUSDT"))
    assert trades
    assert trades[0].pnl_pct > 0
    assert trades[0].exit_reason == "take_profit"


def test_short_stop_loss() -> None:
    df = add_mas(_breakdown_series(after="up"))
    trades = simulate(df, detect_signals(df, "TESTUSDT"))
    assert trades
    assert trades[0].pnl_pct < 0
    assert trades[0].exit_reason == "stop_loss"


def test_no_overlap_positions() -> None:
    df = add_mas(_breakdown_series(n=360, crash=0.90, after="down"))
    close = df["close"].to_numpy(copy=True)
    close[280:300] = close[229] + 0.35
    close[300] = close[229] - 0.90
    for i in range(301, len(close)):
        close[i] = close[i - 1] - 0.12
    df = add_mas(_bars(close))
    df.iloc[230, df.columns.get_loc("high")] = close[229] + 0.08
    df.iloc[300, df.columns.get_loc("high")] = close[299] + 0.08
    sigs = detect_signals(df, "TESTUSDT")
    trades = simulate(df, sigs)
    for a, b in zip(trades, trades[1:]):
        assert a.exit_idx < b.signal.bar_idx


def test_resample_1h_and_snapshot() -> None:
    df = add_mas(_breakdown_series(n=280, after="down"))
    hourly = resample_1h(df)
    assert 68 <= len(hourly) <= 71
    assert "ma7" in hourly.columns
    trades = simulate(df, detect_signals(df, "TESTUSDT"))
    assert trades
    snap = hourly_snapshot(hourly, trades[0].signal.timestamp)
    assert "time" in snap
    assert "ma7" in snap


def test_draw_trade_png_has_hourly(tmp_path: Path | None = None) -> None:
    df = add_mas(_breakdown_series(after="down"))
    trades = simulate(df, detect_signals(df, "TESTUSDT"))
    out = Path("/tmp/ma_break_short_15m_1h.png")
    draw_trade_png(df, trades[0], out, 1, df_1h=resample_1h(df))
    assert out.exists() and out.stat().st_size > 8000


def test_summarize() -> None:
    class T:
        def __init__(self, pnl: float) -> None:
            self.pnl_pct = pnl

    stats = summarize([T(1.5), T(-0.5), T(0.2)])  # type: ignore[list-item]
    assert stats["trades"] == 3
    assert stats["wins"] == 2
    assert abs(stats["total_pnl"] - 1.2) < 1e-9


def main() -> int:
    test_detects_simultaneous_break_and_short_stack()
    test_no_signal_without_short_stack()
    test_short_take_profit()
    test_short_stop_loss()
    test_no_overlap_positions()
    test_resample_1h_and_snapshot()
    test_draw_trade_png_has_hourly()
    test_summarize()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

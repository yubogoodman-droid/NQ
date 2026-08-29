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


def _breakdown_series(n: int = 280, crash: float = 0.65, after: str = "down") -> pd.DataFrame:
    """先緩漲讓長均黏在高位，再回落讓 7<14<25，最後一根大陰線打穿 99/120/200。"""
    close = np.zeros(n, dtype=float)
    close[0] = 20.0
    for i in range(1, 200):
        close[i] = close[i - 1] + 0.012
    peak = close[199]
    for i in range(200, 230):
        close[i] = peak - (i - 199) * 0.018
    break_i = 230
    close[break_i] = close[break_i - 1] - crash
    if after == "down":
        for i in range(break_i + 1, n):
            close[i] = close[i - 1] - 0.10
    else:
        close[break_i + 1] = close[break_i - 1] + 0.40
        for i in range(break_i + 2, n):
            close[i] = close[i - 1] + 0.05
        df = _bars(close)
        df.iloc[break_i, df.columns.get_loc("high")] = close[break_i - 1] + 0.25
        df.iloc[break_i, df.columns.get_loc("open")] = close[break_i - 1]
        df.iloc[break_i, df.columns.get_loc("low")] = close[break_i] - 0.08
        df.iloc[break_i + 1, df.columns.get_loc("high")] = close[break_i + 1] + 0.05
        df.iloc[break_i + 1, df.columns.get_loc("low")] = close[break_i] - 0.02
        return df
    df = _bars(close)
    # 破位那根：高點蓋過長均黏帶上沿，收在三條之下
    df.iloc[break_i, df.columns.get_loc("high")] = close[break_i - 1] + 0.25
    df.iloc[break_i, df.columns.get_loc("open")] = close[break_i - 1]
    df.iloc[break_i, df.columns.get_loc("low")] = close[break_i] - 0.08
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
    df = add_mas(_breakdown_series(n=360, crash=0.65, after="down"))
    # 複製第二次破位：先拉回長均之上再打穿
    close = df["close"].to_numpy(copy=True)
    close[280:300] = close[229] + 0.4
    close[300] = close[229] - 0.70
    for i in range(301, len(close)):
        close[i] = close[i - 1] - 0.10
    df = add_mas(_bars(close))
    df.iloc[230, df.columns.get_loc("high")] = close[229] + 0.25
    df.iloc[300, df.columns.get_loc("high")] = close[299] + 0.25
    sigs = detect_signals(df, "TESTUSDT")
    trades = simulate(df, sigs)
    for a, b in zip(trades, trades[1:]):
        assert a.exit_idx < b.signal.bar_idx


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
    test_summarize()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

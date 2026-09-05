#!/usr/bin/env python3
"""Synthetic tests for NQ 5m 雙底：2h低 → 站上MA20 → 跌破 → 回測守三根。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nq_w_ma20 import (  # noqa: E402
    ET,
    TradeResult,
    detect_signals,
    quality_from_w,
    detect_params,
    resample_ohlc,
    simulate,
    summarize_trades,
    write_html_report,
)


def test_quality_from_w() -> None:
    assert quality_from_w(8.0, 50.0, 6.0)[1] == "A"
    assert quality_from_w(40.0, 16.0, 1.0)[1] == "C"
    assert quality_from_w(8.0, 16.0, 1.0)[1] == "B"


def test_summarize_trades() -> None:
    class T:
        def __init__(self, pnl: float, quality: str = "A"):
            self.pnl_points = pnl
            self.quality = quality

    stats = summarize_trades([T(20.0, "A"), T(-8.0, "B")])
    assert stats["count"] == 2
    assert stats["wins"] == 1
    assert abs(stats["total_points"] - 12.0) < 1e-9


def _ohlc(close: np.ndarray, low: np.ndarray, high: np.ndarray, open_: np.ndarray) -> pd.DataFrame:
    n = len(close)
    idx = pd.date_range("2026-09-04 08:00", periods=n, freq="5min", tz=ET)
    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": np.full(n, 120.0),
        },
        index=idx,
    )


def make_chart29(
    *,
    stand: bool = True,
    break_ma: bool = True,
    retest: bool = True,
    hold3: bool = True,
    deep_spring: bool = False,
) -> pd.DataFrame:
    """對齊圖 29：兩小時低 → 站上 MA20 → 跌破 → 回到低點 → 三根沒新低。"""
    n = 90
    close = np.full(n, 20120.0)
    # 先走一段高位，讓 2h 低在上面
    for i in range(1, 28):
        close[i] = 20120.0 + (i % 5) * 1.5
    # 創兩小時低
    close[28:40] = np.linspace(20110.0, 20040.0, 12)
    if stand:
        close[40:50] = np.linspace(20050.0, 20100.0, 10)
        close[50:56] = 20105.0
    else:
        close[40:] = np.linspace(20042.0, 20020.0, n - 40)
        open_ = np.r_[close[0], close[:-1]]
        high = np.maximum(open_, close) + 2.0
        low = np.minimum(open_, close) - 2.0
        low[39] = 20038.0
        return _ohlc(close, low, high, open_)

    if break_ma:
        close[56:64] = np.linspace(20095.0, 20055.0, 8)
    else:
        close[56:] = 20110.0
        open_ = np.r_[close[0], close[:-1]]
        high = np.maximum(open_, close) + 2.0
        low = np.minimum(open_, close) - 2.0
        low[39] = 20038.0
        high[50] = 20120.0
        return _ohlc(close, low, high, open_)

    if retest:
        close[64:70] = np.linspace(20052.0, 20042.0, 6)
        close[69] = 20044.0
    else:
        # 跌破 MA20 但停在半空，回不到兩小時低點附近
        close[56:60] = [20090.0, 20078.0, 20070.0, 20068.0]
        close[60:] = 20075.0
        open_ = np.r_[close[0], close[:-1]]
        high = np.maximum(open_, close) + 2.0
        low = np.minimum(open_, close) - 2.0
        low[39] = 20038.0
        return _ohlc(close, low, high, open_)

    if hold3:
        close[70:73] = [20048.0, 20050.0, 20052.0]
        close[73:] = np.linspace(20055.0, 20120.0, n - 73)
    else:
        close[70:] = np.linspace(20040.0, 19990.0, n - 70)

    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + 2.5
    low = np.minimum(open_, close) - 2.0
    low[39] = 20038.0
    low[38] = 20045.0
    high[50] = 20118.0
    low[69] = 20036.0  # 回測略破 2h 低（圖 29 那種）
    if hold3:
        low[70] = 20040.0
        low[71] = 20042.0
        low[72] = 20044.0
    if deep_spring:
        low[69] = 20000.0
        close[69] = 20010.0
    if not hold3:
        low[72] = 20010.0
        low[73] = 19995.0
    return _ohlc(close, low, high, open_)


def test_detect_chart29_hold() -> None:
    df = make_chart29()
    sigs = detect_signals(df)
    assert sigs, "圖 29 那種雙底回測守三根應進場"
    sig = sigs[0]
    assert sig.stand_idx > sig.first_low_idx
    assert sig.break_idx > sig.stand_idx
    assert sig.second_low_idx > sig.break_idx
    assert sig.entry_idx > sig.second_low_idx
    assert sig.entry_idx - sig.second_low_idx >= 3
    assert abs(sig.second_low - sig.first_low) <= 20
    assert sig.entry_price > sig.stop_price
    assert sig.target_price > sig.entry_price


def test_no_signal_without_stand() -> None:
    df = make_chart29(stand=False)
    assert not detect_signals(df), "沒站上 MA20 不該進場"


def test_no_signal_without_break() -> None:
    df = make_chart29(break_ma=False)
    assert not detect_signals(df), "站上後沒跌破 MA20 不該進場"


def test_no_signal_without_retest() -> None:
    df = make_chart29(retest=False)
    assert not detect_signals(df), "沒回到兩小時低點附近不該進場"


def test_no_signal_if_keeps_making_lows() -> None:
    df = make_chart29(hold3=False)
    assert not detect_signals(df), "連續三根前又創新低不該進場"


def test_no_signal_on_deep_spring() -> None:
    df = make_chart29(deep_spring=True)
    assert not detect_signals(df), "回測破兩小時低超過 20 點應作廢"


def test_resample_15m() -> None:
    df = make_chart29()
    m15 = resample_ohlc(df, "15min")
    assert len(m15) > 0
    assert len(m15) <= (len(df) + 2) // 3
    assert {"Open", "High", "Low", "Close"} <= set(m15.columns)


def test_simulate_and_html() -> None:
    df = make_chart29()
    sigs = detect_signals(df)
    trades = simulate(df, sigs)
    assert trades
    assert isinstance(trades[0], TradeResult)
    out = Path("/tmp/nq_w_ma20_test.html")
    path = write_html_report(out, df, trades, "NQ=F", "demo")
    text = path.read_text(encoding="utf-8")
    assert "雙底" in text
    assert "MA20" in text
    assert "15分K" in text
    assert "15分K 回測" in text
    assert text.count("<svg") >= 2, "五分圖 + 15 分圖都要內嵌 SVG"


def test_detect_params_15m() -> None:
    assert detect_params("15m")["two_hour_bars"] == 8
    assert detect_params("15m")["max_hold"] == 16
    assert detect_params("5m")["two_hour_bars"] == 24


def main() -> int:
    tests = [
        test_quality_from_w,
        test_summarize_trades,
        test_detect_chart29_hold,
        test_no_signal_without_stand,
        test_no_signal_without_break,
        test_no_signal_without_retest,
        test_no_signal_if_keeps_making_lows,
        test_no_signal_on_deep_spring,
        test_resample_15m,
        test_simulate_and_html,
        test_detect_params_15m,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"ok  {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

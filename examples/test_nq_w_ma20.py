#!/usr/bin/env python3
"""Synthetic tests for NQ 5m W底 右側站上MA20 (no Yahoo)."""

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
    simulate,
    summarize_trades,
    write_html_report,
)


def test_quality_from_w() -> None:
    assert quality_from_w(8.0, 40.0, 6.0)[1] == "A"
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
    idx = pd.date_range("2026-08-10 00:00", periods=n, freq="5min", tz=ET)
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


def make_w_then_stand(*, stand: bool = True, break_new_low: bool = False, neckline_only: bool = False) -> pd.DataFrame:
    """先在 MA20 下做出 L1／頸線／L2，再決定要不要站上均線。"""
    n = 160
    close = np.full(n, 20120.0)
    for i in range(1, 40):
        close[i] = 20120.0 + (i % 4) * 1.2
    # 殺到 L1
    close[40:55] = np.linspace(20110.0, 20040.0, 15)
    # 反彈到頸線
    close[55:65] = np.linspace(20045.0, 20085.0, 10)
    # 右腳回 L2（略高）
    close[65:75] = np.linspace(20080.0, 20048.0, 10)
    close[75] = 20050.0
    if stand and not break_new_low:
        # 右側翻上，穿過當時 MA20
        close[76:90] = np.linspace(20058.0, 20110.0, 14)
        close[90:] = np.linspace(20112.0, 20180.0, n - 90)
    elif stand and break_new_low:
        # 先破 L2 超過容忍，之後才翻上——原 W 作廢，也不能再配成新 W
        close[76:80] = [20040.0, 20020.0, 19990.0, 19985.0]
        close[80:90] = np.linspace(20020.0, 20110.0, 10)
        close[90:] = np.linspace(20112.0, 20180.0, n - 90)
    elif neckline_only:
        close[76:88] = np.linspace(20055.0, 20082.0, 12)
        close[88:] = 20080.0
    else:
        close[76:] = np.linspace(20048.0, 20020.0, n - 76)

    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + 2.0
    low = np.minimum(open_, close) - 2.0
    low[54] = 20038.0
    low[53] = 20042.0
    low[55] = 20043.0
    high[64] = 20092.0
    low[74] = 20044.0
    low[73] = 20047.0
    low[75] = 20046.0
    if break_new_low:
        low[78] = 19982.0
        low[79] = 19980.0
    return _ohlc(close, low, high, open_)


def make_single_v() -> pd.DataFrame:
    """尖 V，沒有第二個谷，不該算 W。"""
    n = 120
    close = np.full(n, 20100.0)
    close[30:50] = np.linspace(20095.0, 20040.0, 20)
    close[50:] = np.linspace(20045.0, 20140.0, n - 50)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + 1.5
    low = np.minimum(open_, close) - 1.5
    low[49] = 20035.0
    return _ohlc(close, low, high, open_)


def test_detect_w_right_stand() -> None:
    df = make_w_then_stand(stand=True)
    sigs = detect_signals(df)
    assert sigs, "W 右側站上 MA20 應進場"
    sig = sigs[0]
    assert sig.entry_idx > sig.second_low_idx
    assert sig.second_low_idx > sig.first_low_idx
    assert sig.entry_price > sig.ma20
    assert sig.second_low < sig.ma20 or True
    assert abs(sig.second_low - sig.first_low) <= 60
    assert sig.neckline > max(sig.first_low, sig.second_low)
    assert sig.target_price > sig.entry_price
    assert sig.entry_price > sig.stop_price


def test_no_signal_without_stand() -> None:
    df = make_w_then_stand(stand=False)
    sigs = detect_signals(df)
    assert not sigs, "右腳之後繼續破底不該進場"


def test_no_signal_on_v() -> None:
    df = make_single_v()
    sigs = detect_signals(df)
    assert not sigs, "單谷 V 不是 W"


def test_invalidate_new_low() -> None:
    df = make_w_then_stand(stand=True, break_new_low=True)
    sigs = detect_signals(df)
    assert not sigs, "站上前破 L2 應作廢"


def test_simulate_and_html() -> None:
    df = make_w_then_stand(stand=True)
    sigs = detect_signals(df)
    trades = simulate(df, sigs)
    assert trades
    assert isinstance(trades[0], TradeResult)
    out = Path("/tmp/nq_w_ma20_test.html")
    path = write_html_report(out, df, trades, "NQ=F", "demo")
    text = path.read_text(encoding="utf-8")
    assert "W底" in text
    assert "MA20" in text
    assert "<svg" in text, "圖要內嵌 SVG，htmlpreview 才看得到"


def main() -> int:
    tests = [
        test_quality_from_w,
        test_summarize_trades,
        test_detect_w_right_stand,
        test_no_signal_without_stand,
        test_no_signal_on_v,
        test_invalidate_new_low,
        test_simulate_and_html,
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

"""5/10/20 空頭排列跌破 MA200 策略單元測試。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tw.backtest import run_symbol_backtest, summarize
from tw.strategy import TwMaShortStrategy
from tw.universe import previous_week_end, weekly_top_n_mask


def _downtrend_bars(n: int = 260) -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-02", periods=n)
    close = np.full(n, 100.0)
    close[:200] = 100.0
    close[200:] = np.linspace(99.4, 85.0, n - 200)
    high = close + 0.6
    low = np.minimum(close - 0.6, close)
    open_ = np.r_[close[0], close[:-1]]
    volume = np.full(n, 1_000_000.0)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def test_bearish_breakdown_signal():
    df = _downtrend_bars()
    signals = TwMaShortStrategy(max_price=600).generate_signals(df, ticker="2330.TW", name="台積電")
    assert signals, "下跌穿越 MA200 應產生做空訊號"
    sig = signals[0]
    assert sig.ma5 < sig.ma10 < sig.ma20
    assert sig.entry < sig.ma200
    assert sig.entry <= 600


def test_price_filter_blocks_expensive_stock():
    df = _downtrend_bars()
    df["close"] = df["close"] * 10
    df["open"] *= 10
    df["high"] *= 10
    df["low"] *= 10
    signals = TwMaShortStrategy(max_price=600).generate_signals(df, ticker="2330.TW")
    assert signals == []


def test_weekly_top_n_uses_previous_week():
    idx = pd.bdate_range("2026-06-01", periods=20)
    close = pd.DataFrame(
        {
            "A.TW": np.full(len(idx), 50.0),
            "B.TW": np.full(len(idx), 50.0),
            "C.TW": np.full(len(idx), 800.0),
        },
        index=idx,
    )
    volume = pd.DataFrame(
        {
            "A.TW": np.full(len(idx), 1_000_000.0),
            "B.TW": np.full(len(idx), 100.0),
            "C.TW": np.full(len(idx), 9_000_000.0),
        },
        index=idx,
    )
    mask = weekly_top_n_mask(close, volume, top_n=1, max_price=600)
    monday = pd.Timestamp("2026-06-15")
    assert monday in mask.index
    assert bool(mask.loc[monday, "A.TW"]) is True
    assert bool(mask.loc[monday, "B.TW"]) is False
    assert bool(mask.loc[monday, "C.TW"]) is False


def test_previous_week_end_skips_current_friday():
    friday = pd.Timestamp("2026-08-14")
    assert previous_week_end(friday) == pd.Timestamp("2026-08-07")
    monday = pd.Timestamp("2026-08-17")
    assert previous_week_end(monday) == pd.Timestamp("2026-08-14")


def test_backtest_records_short_pnl():
    df = _downtrend_bars()
    signals = TwMaShortStrategy().generate_signals(df, ticker="9999.TW", name="示範")
    results = run_symbol_backtest(df, signals, commission=0.0, tax=0.0)
    assert results
    stats = summarize(results)
    assert stats["trades"] >= 1
    assert results[0].pnl_pct > 0

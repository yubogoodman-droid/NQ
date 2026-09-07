#!/usr/bin/env python3
"""Synthetic tests for 牛来 5m M頭跌破 MA200（不打幣安）。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from niulai_m_top import (  # noqa: E402
    CST,
    detect_m_tops,
    default_params,
    display_name,
    filter_entry_window,
    fmt_px,
    generate_signals,
    rank_usdt_perps,
    simulate,
    sma,
    summarize_trades,
)


def _idx(n: int, start: str = "2026-09-01 10:00") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="5min", tz=CST)


def _ohlc(close: np.ndarray, high_extra: float = 0.0004, low_extra: float = 0.0004) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    n = len(close)
    opens = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(opens, close) + high_extra
    low = np.minimum(opens, close) - low_extra
    return pd.DataFrame(
        {"open": opens, "high": high, "low": low, "close": close, "volume": np.full(n, 1000.0)},
        index=_idx(n),
    )


def test_sma() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
    out = sma(x, 3)
    assert np.isnan(out[1])
    assert abs(out[2] - 2.0) < 1e-9
    assert abs(out[4] - 4.0) < 1e-9


def _m_top_series(n: int = 320) -> tuple[np.ndarray, dict[str, int]]:
    """先在 MA200 上方盤，做出 H1 / 頸線 / H2，再收盤跌破 MA200。"""
    close = np.full(n, 0.1000, dtype=float)
    # 讓 MA200 落在 ~0.100，後面用更高的峰再跌破
    for i in range(1, 210):
        close[i] = 0.1000 + 0.00002 * np.sin(i / 9.0)

    h1 = 230
    neck = 248
    h2 = 266
    br = 278
    close[210:h1] = np.linspace(0.1004, 0.1060, h1 - 210)
    close[h1] = 0.1062
    close[h1 + 1 : neck] = np.linspace(0.1056, 0.1024, neck - h1 - 1)
    close[neck] = 0.1022
    close[neck + 1 : h2] = np.linspace(0.1026, 0.1058, h2 - neck - 1)
    close[h2] = 0.1060
    close[h2 + 1 : br] = np.linspace(0.1054, 0.1012, br - h2 - 1)
    close[br] = 0.0994  # 明確跌破 ~0.100 的 MA200
    for i in range(br + 1, n):
        close[i] = close[i - 1] - 0.00035
    marks = {"h1": h1, "neck": neck, "h2": h2, "br": br}
    return close, marks


def test_detects_m_top_ma200_short() -> None:
    close, marks = _m_top_series()
    df = _ohlc(close)
    # 把 H1/H2 的 high 做成真正的波段高
    df.loc[df.index[marks["h1"]], "high"] = 0.1068
    df.loc[df.index[marks["h2"]], "high"] = 0.1066
    df.loc[df.index[marks["neck"]], "low"] = 0.1016
    params = default_params(min_depth_pct=0.015, high_tolerance_pct=0.03)
    patterns = detect_m_tops(df, params)
    assert patterns, "should find an M-top"
    p = patterns[-1]
    assert abs(p.first_high_idx - marks["h1"]) <= 2
    assert abs(p.second_high_idx - marks["h2"]) <= 2
    assert p.breakout_idx >= marks["h2"]
    sigs = generate_signals(df, params)
    assert sigs
    assert sigs[-1].stop_loss > sigs[-1].entry
    assert sigs[-1].target < sigs[-1].entry


def test_rejects_higher_high_between_peaks() -> None:
    close, marks = _m_top_series()
    df = _ohlc(close)
    df.loc[df.index[marks["h1"]], "high"] = 0.1068
    df.loc[df.index[marks["h2"]], "high"] = 0.1066
    mid = (marks["h1"] + marks["h2"]) // 2
    df.loc[df.index[mid], "high"] = 0.1090  # 兩峰之間更高 → 不是 M
    params = default_params(min_depth_pct=0.015)
    patterns = detect_m_tops(df, params)
    assert all(not (abs(p.first_high_idx - marks["h1"]) <= 2 and abs(p.second_high_idx - marks["h2"]) <= 2) for p in patterns)


def test_simulate_target_and_stop() -> None:
    close, marks = _m_top_series(360)
    df = _ohlc(close)
    df.loc[df.index[marks["h1"]], "high"] = 0.1068
    df.loc[df.index[marks["h2"]], "high"] = 0.1066
    df.loc[df.index[marks["neck"]], "low"] = 0.1016
    params = default_params(min_depth_pct=0.015, time_bars=80, target_r=2.0)
    sigs = generate_signals(df, params)
    assert sigs
    # 目標：把進場後的低點打到 2R
    trades = simulate(df, sigs, params)
    assert trades
    t = trades[-1]
    assert t.exit_reason in {"target", "time", "stop"}
    if t.exit_reason == "target":
        assert t.pnl_pct > 0
        assert abs(t.exit_price - t.target_price) < 1e-9

    # 停損：進場後立刻創 M 頭新高
    df2 = df.copy()
    stop_bar = t.entry_idx + 2
    df2.loc[df2.index[stop_bar], "high"] = t.stop_price + 0.001
    trades2 = simulate(df2, sigs, params)
    assert trades2[-1].exit_reason == "stop"
    assert trades2[-1].pnl_pct < 0


def test_one_position_at_a_time() -> None:
    close, marks = _m_top_series(400)
    df = _ohlc(close)
    df.loc[df.index[marks["h1"]], "high"] = 0.1068
    df.loc[df.index[marks["h2"]], "high"] = 0.1066
    df.loc[df.index[marks["neck"]], "low"] = 0.1016
    params = default_params(min_depth_pct=0.015, time_bars=80)
    sigs = generate_signals(df, params)
    if len(sigs) < 2:
        # 複製一筆較晚的假訊號，確認持倉中會跳過
        late = sigs[-1]
        from dataclasses import replace

        fake = replace(late, bar_idx=late.bar_idx + 3, timestamp=df.index[late.bar_idx + 3])
        sigs = list(sigs) + [fake]
    trades = simulate(df, sigs, params)
    for a, b in zip(trades, trades[1:]):
        assert b.entry_idx > a.exit_idx


def test_filter_entry_window() -> None:
    close = np.linspace(0.10, 0.09, 40)
    df = _ohlc(close)
    from niulai_m_top import MTopPattern, Signal

    p = MTopPattern(0, 10, 5, 0.11, 0.11, 0.10, 12)
    early = Signal(df.index[2], 0.10, 0.11, 0.08, p, 2, 0.105)
    late = Signal(df.index[-2], 0.09, 0.11, 0.07, p, len(df) - 2, 0.095)
    kept = filter_entry_window(df, [early, late], days=1)
    assert late in kept
    assert early not in kept or (df.index[-1] - early.timestamp) <= pd.Timedelta(days=1)


def test_summarize() -> None:
    from niulai_m_top import MTopPattern, Signal, TradeResult

    p = MTopPattern(0, 1, 0, 1.0, 1.0, 0.9, 2)

    def _t(pnl: float, reason: str) -> TradeResult:
        sig = Signal(pd.Timestamp("2026-09-07", tz=CST), 1.0, 1.1, 0.8, p, 2, 1.0)
        return TradeResult(sig, 2, 5, 1.0, 1.0 - pnl, 1.1, 0.8, pnl, reason)

    stats = summarize_trades([_t(0.05, "target"), _t(-0.02, "stop"), _t(0.01, "time")])
    assert stats["count"] == 3
    assert stats["wins"] == 2
    assert abs(stats["total_pct"] - 0.04) < 1e-9
    assert stats["reasons"]["target"] == 1


def test_rank_usdt_perps_top50() -> None:
    symbols = [
        {
            "symbol": "BTCUSDT",
            "quoteAsset": "USDT",
            "status": "TRADING",
            "contractType": "PERPETUAL",
            "underlyingType": "COIN",
            "baseAsset": "BTC",
            "filters": [{"filterType": "PRICE_FILTER", "tickSize": "0.10"}],
        },
        {
            "symbol": "ETHUSDT",
            "quoteAsset": "USDT",
            "status": "TRADING",
            "contractType": "PERPETUAL",
            "underlyingType": "COIN",
            "baseAsset": "ETH",
            "filters": [{"filterType": "PRICE_FILTER", "tickSize": "0.01"}],
        },
        {
            "symbol": "TSLAUSDT",
            "quoteAsset": "USDT",
            "status": "TRADING",
            "contractType": "TRADIFI_PERPETUAL",
            "underlyingType": "COIN",
            "baseAsset": "TSLA",
            "filters": [],
        },
        {
            "symbol": "DEADUSDT",
            "quoteAsset": "USDT",
            "status": "SETTLING",
            "contractType": "PERPETUAL",
            "underlyingType": "COIN",
            "baseAsset": "DEAD",
            "filters": [],
        },
        {
            "symbol": "牛来USDT",
            "quoteAsset": "USDT",
            "status": "TRADING",
            "contractType": "PERPETUAL",
            "underlyingType": "COIN",
            "baseAsset": "牛来",
            "filters": [{"filterType": "PRICE_FILTER", "tickSize": "0.0000100"}],
        },
    ]
    tickers = [
        {"symbol": "BTCUSDT", "quoteVolume": "100", "lastPrice": "70000"},
        {"symbol": "ETHUSDT", "quoteVolume": "300", "lastPrice": "3000"},
        {"symbol": "TSLAUSDT", "quoteVolume": "9999", "lastPrice": "400"},
        {"symbol": "DEADUSDT", "quoteVolume": "800", "lastPrice": "1"},
        {"symbol": "牛来USDT", "quoteVolume": "200", "lastPrice": "0.08"},
    ]
    rows = rank_usdt_perps(symbols, tickers, limit=50)
    assert [r.symbol for r in rows] == ["ETHUSDT", "牛来USDT", "BTCUSDT"]
    assert rows[0].rank == 1
    assert rows[1].base == "牛来"
    assert abs(rows[2].tick_size - 0.10) < 1e-9
    tiny = rank_usdt_perps(symbols, tickers, limit=2)
    assert len(tiny) == 2
    assert tiny[0].symbol == "ETHUSDT"


def test_display_and_fmt() -> None:
    assert display_name("BTCUSDT") == "BTC"
    assert display_name("牛来USDT") == "牛来"
    assert fmt_px(70123.4) == "70123.40"
    assert fmt_px(0.08978) == "0.08978"


def main() -> int:
    tests = [
        test_sma,
        test_detects_m_top_ma200_short,
        test_rejects_higher_high_between_peaks,
        test_simulate_target_and_stop,
        test_one_position_at_a_time,
        test_filter_entry_window,
        test_summarize,
        test_rank_usdt_perps_top50,
        test_display_and_fmt,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"ok  {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
            import traceback

            traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

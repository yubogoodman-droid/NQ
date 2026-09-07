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
    _idx_at,
    detect_m_tops,
    default_params,
    display_name,
    draw_hourly_png,
    filter_entry_window,
    fmt_px,
    generate_signals,
    niulai_params,
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


def _stamp_m(
    df: pd.DataFrame,
    marks: dict[str, int],
    h1: float,
    h2: float,
    neck: float,
    look: int = 3,
) -> pd.DataFrame:
    """把 H1/H2 做成唯一波段高，避免鄰棒 wick 打平。"""
    i1, i2, nk = marks["h1"], marks["h2"], marks["neck"]
    df.loc[df.index[i1], "high"] = h1
    df.loc[df.index[i2], "high"] = h2
    df.loc[df.index[nk], "low"] = neck
    hi = df.columns.get_loc("high")
    for i, peak in ((i1, h1), (i2, h2)):
        for j in range(i - look, i + look + 1):
            if j == i or j < 0 or j >= len(df):
                continue
            df.iloc[j, hi] = min(float(df.iloc[j, hi]), peak - 0.00035)
    return df


def _niulai_like_series(n: int = 330) -> tuple[np.ndarray, dict[str, int]]:
    """截圖同款：雙峰等高、頸線回到 MA200、深度約 5%、峰在均線上方。"""
    close = np.full(n, 0.1000, dtype=float)
    for i in range(1, 210):
        close[i] = 0.1000 + 0.00002 * np.sin(i / 9.0)
    h1, neck, h2, br = 236, 256, 276, 290
    close[210:h1] = np.linspace(0.1003, 0.1046, h1 - 210)
    close[h1] = 0.1048
    close[h1 + 1 : neck] = np.linspace(0.1042, 0.1002, neck - h1 - 1)
    close[neck] = 0.0996
    close[neck + 1 : h2] = np.linspace(0.1001, 0.1044, h2 - neck - 1)
    close[h2] = 0.1046
    close[h2 + 1 : br] = np.linspace(0.1040, 0.1008, br - h2 - 1)
    close[br] = 0.0988
    for i in range(br + 1, n):
        close[i] = close[i - 1] - 0.00040
    return close, {"h1": h1, "neck": neck, "h2": h2, "br": br}


def test_niulai_params_keeps_screenshot_shape() -> None:
    close, marks = _niulai_like_series()
    df = _stamp_m(_ohlc(close), marks, 0.1052, 0.1050, 0.0994)
    patterns = detect_m_tops(df, niulai_params())
    assert patterns, "牛來型參數應抓到對稱深 M"
    p = patterns[-1]
    assert abs(p.first_high_idx - marks["h1"]) <= 2
    assert abs(p.second_high_idx - marks["h2"]) <= 2
    assert p.depth_pct >= 0.04
    assert p.breakout_idx >= marks["h2"]


def _shallow_chop_series(n: int = 320) -> tuple[np.ndarray, dict[str, int]]:
    """SOPH 那種貼均線的小 M：深度 ~2%、峰只比 MA200 高一點。"""
    close = np.full(n, 0.1000, dtype=float)
    for i in range(1, 210):
        close[i] = 0.1000 + 0.00002 * np.sin(i / 9.0)
    h1, neck, h2, br = 230, 248, 266, 278
    close[210:h1] = np.linspace(0.1002, 0.1015, h1 - 210)
    close[h1] = 0.1016
    close[h1 + 1 : neck] = np.linspace(0.1014, 0.1004, neck - h1 - 1)
    close[neck] = 0.1003
    close[neck + 1 : h2] = np.linspace(0.1005, 0.1014, h2 - neck - 1)
    close[h2] = 0.1015
    close[h2 + 1 : br] = np.linspace(0.1013, 0.1004, br - h2 - 1)
    close[br] = 0.0996
    for i in range(br + 1, n):
        close[i] = 0.1000 + 0.00005 * np.sin(i / 7.0)
    return close, {"h1": h1, "neck": neck, "h2": h2, "br": br}


def test_niulai_params_rejects_shallow_chop() -> None:
    """SOPH 那種貼均線、深度只有 2% 的假 M，不該進場。"""
    close, marks = _shallow_chop_series()
    df = _stamp_m(_ohlc(close), marks, 0.1020, 0.1019, 0.1002)
    assert detect_m_tops(df, niulai_params()) == []


def test_niulai_params_rejects_higher_right_peak() -> None:
    close, marks = _niulai_like_series()
    df = _stamp_m(_ohlc(close), marks, 0.1052, 0.1064, 0.0994)
    patterns = detect_m_tops(df, niulai_params())
    assert all(
        not (abs(p.first_high_idx - marks["h1"]) <= 2 and abs(p.second_high_idx - marks["h2"]) <= 2)
        for p in patterns
    )


def test_niulai_params_rejects_neck_far_below_ma() -> None:
    """頸線深跌穿 MA200（USELESS / BULLA 那種）不是牛來。"""
    close, marks = _niulai_like_series()
    df = _stamp_m(_ohlc(close), marks, 0.1052, 0.1050, 0.0950)
    patterns = detect_m_tops(df, niulai_params())
    assert all(
        not (abs(p.first_high_idx - marks["h1"]) <= 2 and abs(p.second_high_idx - marks["h2"]) <= 2)
        for p in patterns
    )


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


def test_idx_at_maps_intrabar_to_hourly() -> None:
    idx = pd.date_range("2026-09-07 18:00", periods=4, freq="h", tz=CST)
    df = pd.DataFrame({"close": [1.0, 1.1, 1.2, 1.3]}, index=idx)
    i = _idx_at(df, pd.Timestamp("2026-09-07 18:10", tz=CST))
    assert i == 0
    assert df.index[i] == idx[0]
    assert _idx_at(df, pd.Timestamp("2026-09-07 17:00", tz=CST)) is None
    assert _idx_at(pd.DataFrame(), pd.Timestamp("2026-09-07 18:10", tz=CST)) is None


def test_draw_hourly_png_for_5m_trade() -> None:
    import tempfile

    close, marks = _m_top_series(360)
    df = _ohlc(close)
    df.loc[df.index[marks["h1"]], "high"] = 0.1068
    df.loc[df.index[marks["h2"]], "high"] = 0.1066
    df.loc[df.index[marks["neck"]], "low"] = 0.1016
    params = default_params(min_depth_pct=0.015, time_bars=80, target_r=2.0)
    trades = simulate(df, generate_signals(df, params), params)
    assert trades
    hourly = (
        df.resample("1h")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
    )
    pad_idx = pd.date_range(
        df.index[0] - pd.Timedelta(hours=20),
        hourly.index[0],
        freq="h",
        tz=CST,
        inclusive="left",
    )
    pad = pd.DataFrame(
        {"open": 0.10, "high": 0.101, "low": 0.099, "close": 0.10, "volume": 1000.0},
        index=pad_idx,
    )
    tail_idx = pd.date_range(
        hourly.index[-1] + pd.Timedelta(hours=1),
        df.index[-1] + pd.Timedelta(hours=10),
        freq="h",
        tz=CST,
    )
    tail = pd.DataFrame(
        {"open": 0.09, "high": 0.091, "low": 0.089, "close": 0.09, "volume": 1000.0},
        index=tail_idx,
    )
    hourly = pd.concat([pad, hourly, tail]).sort_index()
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "trade_1h.png"
        out = draw_hourly_png(hourly, df, trades[-1], path, 1)
        assert out is not None
        assert path.is_file() and path.stat().st_size > 1000


def test_display_and_fmt() -> None:
    assert display_name("BTCUSDT") == "BTC"
    assert display_name("牛来USDT") == "牛来"
    assert fmt_px(70123.4) == "70123.40"
    assert fmt_px(0.08978) == "0.08978"


def main() -> int:
    tests = [
        test_sma,
        test_niulai_params_keeps_screenshot_shape,
        test_niulai_params_rejects_shallow_chop,
        test_niulai_params_rejects_higher_right_peak,
        test_niulai_params_rejects_neck_far_below_ma,
        test_detects_m_top_ma200_short,
        test_rejects_higher_high_between_peaks,
        test_simulate_target_and_stop,
        test_one_position_at_a_time,
        test_filter_entry_window,
        test_summarize,
        test_rank_usdt_perps_top50,
        test_idx_at_maps_intrabar_to_hourly,
        test_draw_hourly_png_for_5m_trade,
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

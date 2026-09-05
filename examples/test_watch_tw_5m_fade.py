#!/usr/bin/env python3
"""Synthetic tests for TW 5m 5/10/20 空排跌破 MA240 (no network)."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scan_tw_ma_reclaim import TPE  # noqa: E402
from watch_tw_5m_fade import (  # noqa: E402
    detect_signals,
    drop_incomplete_5m,
    fmt_alert,
    hit_on_day,
    hit_within_max_price,
    in_tw_session,
    merge_universe,
    parse_symbols,
    ribbon_down,
    signal_key,
    simulate,
    sma,
    summarize_trades,
    write_html_report,
)


def _session_index(n: int, start: str = "2026-08-17 09:00") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="5min", tz=TPE)


def _ohlc_from_close(close: np.ndarray) -> pd.DataFrame:
    high = close + 0.25
    low = close - 0.25
    idx = _session_index(len(close))
    return pd.DataFrame(
        {
            "Open": np.r_[close[0], close[:-1]],
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": np.full(len(close), 1200.0),
        },
        index=idx,
    )


def make_ma240_break_bars(n: int = 280) -> pd.DataFrame:
    """緩升站在年線上，最後急殺讓收盤剛跌破 MA240，且 5<10<20 下彎。"""
    climb_n = n - 6
    close = np.linspace(90.0, 118.0, climb_n)
    dump = np.array([116.5, 112.0, 107.5, 104.0, 101.5, 100.0])
    close = np.r_[close, dump]
    df = _ohlc_from_close(close)
    df.loc[df.index[-5], "Volume"] = 4800.0
    return df


def make_flat_above_ma240(n: int = 260) -> pd.DataFrame:
    close = 100.0 + np.linspace(0, 1.2, n)
    return _ohlc_from_close(close)


def test_sma() -> None:
    out = sma(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 3)
    assert np.isnan(out[1])
    assert abs(out[2] - 2.0) < 1e-9


def test_parse_symbols() -> None:
    rows = parse_symbols("2609, 2330.TW, 6488.TWO")
    assert [r["symbol"] for r in rows] == ["2609.TW", "2330.TW", "6488.TWO"]
    assert rows[0]["code"] == "2609"
    assert rows[2]["market"] == "otc"


def test_ribbon_down() -> None:
    ma5 = np.array([10.0, 9.5, 9.0])
    ma10 = np.array([10.2, 9.8, 9.4])
    ma20 = np.array([10.4, 10.0, 9.7])
    assert ribbon_down(ma5, ma10, ma20, 2)
    assert not ribbon_down(ma20, ma10, ma5, 2)  # 多排
    rising5 = np.array([9.0, 9.2, 9.4])
    rising10 = np.array([9.3, 9.5, 9.7])
    rising20 = np.array([9.6, 9.8, 10.0])
    assert not ribbon_down(rising5, rising10, rising20, 2)
    assert ribbon_down(rising5, rising10, rising20, 2, require_falling=False)
    falling_glue = np.array([10.0, 9.99, 9.98])
    assert ribbon_down(falling_glue, falling_glue + 0.02, falling_glue + 0.04, 2)


def test_detect_ma240_break() -> None:
    df = make_ma240_break_bars()
    sigs = detect_signals(df)
    assert sigs, "緩升後急殺跌破 MA240 應出訊號"
    sig = sigs[0]
    assert sig.entry_idx == sig.break_idx
    assert sig.entry_price < sig.ma240
    ma240s = sma(df["Close"].to_numpy(float), 240)
    assert sig.prev_close >= float(ma240s[sig.entry_idx - 1])
    assert sig.ma5 < sig.ma10 < sig.ma20
    assert sig.entry_price < sig.ma5


def test_yangming_like_open_dump_not_skipped() -> None:
    """開盤後不久跌破也要抓（陽明 09:05），不要擋 09:30 前。"""
    df = make_ma240_break_bars()
    # 把跌破那根改到 09:05
    sigs = detect_signals(df)
    assert sigs
    # 預設 skip_before=None，即使訊號在早上也在
    early = detect_signals(df, skip_before=(9, 30))
    late = detect_signals(df, skip_before=None)
    assert len(late) >= len(early)


def test_flat_market_has_no_signal() -> None:
    assert detect_signals(make_flat_above_ma240()) == []


def test_skip_before_filters_all_morning() -> None:
    df = make_ma240_break_bars()
    assert detect_signals(df)
    assert detect_signals(df, skip_before=(23, 59)) == []


def test_drop_incomplete_5m() -> None:
    idx = pd.DatetimeIndex(
        [
            "2026-08-25 11:15:00",
            "2026-08-25 11:20:00",
            "2026-08-25 11:21:27",
        ],
        tz=TPE,
    )
    df = pd.DataFrame(
        {"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": 1.0},
        index=idx,
    )
    now = datetime(2026, 8, 25, 11, 21, 30, tzinfo=TPE)
    out = drop_incomplete_5m(df, now=now)
    assert list(out.index) == [idx[0]]


def test_simulate_short_and_summarize() -> None:
    df = make_ma240_break_bars()
    sigs = detect_signals(df)
    trades = simulate(df, sigs)
    assert trades
    t = trades[0]
    assert t.stop_price > t.entry_price
    assert t.target_price < t.entry_price
    assert t.exit_reason in {"stop", "target", "eod"}
    stats = summarize_trades(trades)
    assert stats["count"] == len(trades)


def test_signal_key_and_alert_text() -> None:
    df = make_ma240_break_bars()
    sig = detect_signals(df)[0]
    row = {"code": "2609", "name": "陽明", "symbol": "2609.TW"}
    key = signal_key(row, df, sig)
    assert key.startswith("2609.TW|")
    text = fmt_alert(row, df, sig)
    assert "2609" in text
    assert "MA240" in text
    assert "空頭排列" in text


def test_merge_universe_and_day_filter() -> None:
    base = parse_symbols("2330")
    extra = parse_symbols("2609,2330")
    merged = merge_universe(base, extra)
    assert [r["code"] for r in merged] == ["2330", "2609"]
    df = make_ma240_break_bars()
    sig = detect_signals(df)[0]
    assert hit_on_day(df, sig, df.index[sig.entry_idx].date())
    assert not hit_on_day(df, sig, datetime(2026, 1, 1).date())
    row = {"code": "2609", "close": 57.6}
    assert hit_within_max_price(row, sig, df, 400.0)
    assert not hit_within_max_price(row, sig, df, 50.0)


def test_in_tw_session() -> None:
    lunch = datetime(2026, 8, 21, 11, 30, tzinfo=TPE)
    sunday = datetime(2026, 8, 23, 11, 30, tzinfo=TPE)
    night = datetime(2026, 8, 21, 20, 0, tzinfo=TPE)
    assert in_tw_session(lunch)
    assert not in_tw_session(sunday)
    assert not in_tw_session(night)


def test_write_html(tmp_path: Path | None = None) -> None:
    df = make_ma240_break_bars()
    sigs = detect_signals(df)
    trades = simulate(df, sigs)
    out_dir = tmp_path or Path("/tmp/tw5m_ma240_short_test")
    html = out_dir / "index.html"
    row = {"code": "2609", "name": "陽明", "symbol": "2609.TW"}
    hits = [(row, sigs[0], trades[0], df)]
    path = write_html_report(html, hits, [row], "7d")
    text = path.read_text(encoding="utf-8")
    assert "MA240" in text
    assert "2609" in text
    assert "空頭排列" in text
    assert (path.parent / "img").exists()


def main() -> int:
    test_sma()
    test_parse_symbols()
    test_ribbon_down()
    test_detect_ma240_break()
    test_yangming_like_open_dump_not_skipped()
    test_flat_market_has_no_signal()
    test_skip_before_filters_all_morning()
    test_drop_incomplete_5m()
    test_simulate_short_and_summarize()
    test_signal_key_and_alert_text()
    test_merge_universe_and_day_filter()
    test_in_tw_session()
    test_write_html()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

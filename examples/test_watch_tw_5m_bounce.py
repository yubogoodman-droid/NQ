#!/usr/bin/env python3
"""Synthetic tests for TW 5m 破底反彈 → 5/10/20 多頭排列 (no network)."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scan_tw_ma_reclaim import TPE  # noqa: E402
from watch_tw_5m_bounce import (  # noqa: E402
    detect_signals,
    drop_incomplete_5m,
    fmt_alert,
    hit_on_day,
    hit_within_max_price,
    in_tw_session,
    merge_universe,
    parse_symbols,
    signal_key,
    simulate,
    sma,
    stack_pretty,
    summarize_trades,
    write_html_report,
)


def _session_index(n: int, start: str = "2026-08-21 09:00") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="5min", tz=TPE)


def _ohlc(close: np.ndarray, dump_i: int, trough: float) -> pd.DataFrame:
    high = close + 0.35
    low = close - 0.35
    low[dump_i] = trough
    high[dump_i] = max(close[dump_i], trough) + 0.2
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


def make_v_bounce_bars(n: int = 90) -> pd.DataFrame:
    """高原 272 → 急殺 259 → 拉回站上均線，讓 5/10/20 重新多頭排列。"""
    close = np.full(n, 272.0, dtype=float)
    for i in range(1, 52):
        close[i] = 271.6 + (i % 5) * 0.15
    dump_i = 58
    close[52:dump_i] = np.linspace(271.8, 266.0, dump_i - 52, endpoint=False)
    close[dump_i] = 261.0
    close[dump_i + 1] = 263.5
    close[dump_i + 2] = 266.0
    close[dump_i + 3] = 268.2
    close[dump_i + 4] = 270.0
    close[dump_i + 5] = 271.4
    close[dump_i + 6] = 272.6
    close[dump_i + 7] = 273.4
    for i in range(dump_i + 8, n):
        close[i] = 273.2 + (i - dump_i - 8) * 0.12
    df = _ohlc(close, dump_i, 259.0)
    df.loc[df.index[dump_i], "Volume"] = 4800.0
    df.loc[df.index[dump_i + 6], "Volume"] = 3600.0
    return df


def make_tangled_bounce_bars(n: int = 90) -> pd.DataFrame:
    """急殺後在均線附近橫盤磨，5/10/20 黏在一起。"""
    close = np.full(n, 212.0, dtype=float)
    for i in range(1, 52):
        close[i] = 211.8 + (i % 3) * 0.04
    dump_i = 58
    close[52:dump_i] = np.linspace(211.8, 209.4, dump_i - 52, endpoint=False)
    close[dump_i] = 209.0
    wobble = np.array([0.10, -0.08, 0.06, -0.05, 0.08, -0.07, 0.05, -0.04, 0.07, -0.06, 0.04, -0.03])
    for j, i in enumerate(range(dump_i + 1, n)):
        close[i] = 211.15 + wobble[j % len(wobble)]
    return _ohlc(close, dump_i, 208.9)


def make_flat_bars(n: int = 80) -> pd.DataFrame:
    close = 100.0 + np.linspace(0, 0.8, n)
    high = close + 0.15
    low = close - 0.15
    idx = _session_index(n)
    return pd.DataFrame(
        {
            "Open": np.r_[close[0], close[:-1]],
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": np.full(n, 800.0),
        },
        index=idx,
    )


def test_sma() -> None:
    out = sma(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 3)
    assert np.isnan(out[1])
    assert abs(out[2] - 2.0) < 1e-9


def test_parse_symbols() -> None:
    rows = parse_symbols("6239, 2330.TW, 6488.TWO")
    assert [r["symbol"] for r in rows] == ["6239.TW", "2330.TW", "6488.TWO"]
    assert rows[0]["code"] == "6239"
    assert rows[2]["market"] == "otc"


def test_merge_universe_and_day_filter() -> None:
    base = parse_symbols("2330")
    extra = parse_symbols("6239,2330")
    merged = merge_universe(base, extra)
    assert [r["code"] for r in merged] == ["2330", "6239"]
    df = make_v_bounce_bars()
    sig = detect_signals(df)[0]
    assert hit_on_day(df, sig, df.index[sig.entry_idx].date())
    assert not hit_on_day(df, sig, datetime(2026, 1, 1).date())
    row = {"code": "6239", "close": 269.0}
    assert hit_within_max_price(row, sig, df, 400.0)
    assert not hit_within_max_price(row, sig, df, 100.0)
    expensive = {"code": "2327", "close": 590.0}
    assert not hit_within_max_price(expensive, sig, df, 400.0)
    prev = df.iloc[[0]].copy()
    prev.index = prev.index - pd.Timedelta(days=1)
    prev["High"] = 900.0
    week = pd.concat([prev, df])
    assert hit_within_max_price(row, detect_signals(week)[0], week, 400.0)


def test_detect_v_bounce_like_6239() -> None:
    df = make_v_bounce_bars()
    sigs = detect_signals(df)
    assert sigs, "急殺 272→259 再拉回後應出現 5/10/20 多頭排列"
    sig = sigs[0]
    assert sig.entry_idx > sig.break_idx
    assert sig.break_low <= 259.01
    assert sig.drop_pct >= 0.02
    assert sig.bounce_pct >= 0.015
    assert sig.ma5 > sig.ma10 > sig.ma20
    assert sig.entry_idx - sig.break_idx <= 24


def test_skip_before_filters_open() -> None:
    df = make_v_bounce_bars()
    assert detect_signals(df)
    assert detect_signals(df, skip_before=(23, 59)) == []


def test_flat_market_has_no_signal() -> None:
    assert detect_signals(make_flat_bars()) == []


def test_tangled_ribbon_skipped() -> None:
    df = make_tangled_bounce_bars()
    pretty = detect_signals(df, require_pretty=True)
    loose = detect_signals(df, require_pretty=False)
    assert pretty == []
    assert len(pretty) <= len(loose)


def test_stack_pretty_rejects_glued_mas() -> None:
    n = 30
    ma5 = np.linspace(100.02, 100.08, n)
    ma10 = ma5 - 0.01
    ma20 = ma10 - 0.01
    close = ma5 + 0.02
    assert not stack_pretty(ma5, ma10, ma20, close, n - 1)
    ma5b = np.array([100.0 + i * 0.35 for i in range(n)])
    ma10b = np.array([99.6 + i * 0.22 for i in range(n)])
    ma20b = np.array([99.3 + i * 0.12 for i in range(n)])
    closeb = ma5b + 0.4
    assert stack_pretty(ma5b, ma10b, ma20b, closeb, n - 1)


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


def test_shallow_dip_ignored() -> None:
    close = np.full(80, 100.0)
    close[50] = 99.4  # 0.6%，低於 2% 門檻
    df = _ohlc(close, 50, 99.3)
    assert detect_signals(df) == []


def test_simulate_and_summarize() -> None:
    df = make_v_bounce_bars()
    sigs = detect_signals(df)
    trades = simulate(df, sigs)
    assert trades
    stats = summarize_trades(trades)
    assert stats["count"] == len(trades)
    t = trades[0]
    assert t.exit_idx >= t.entry_idx
    assert t.exit_reason in {"stop", "target", "eod"}


def test_signal_key_and_alert_text() -> None:
    df = make_v_bounce_bars()
    sig = detect_signals(df)[0]
    row = {"code": "6239", "name": "力成", "symbol": "6239.TW"}
    key = signal_key(row, df, sig)
    assert key.startswith("6239.TW|")
    text = fmt_alert(row, df, sig)
    assert "6239" in text
    assert "多頭排列" in text
    assert "MA5" in text
    assert "間隔" in text


def test_in_tw_session() -> None:
    lunch = datetime(2026, 8, 21, 11, 30, tzinfo=TPE)
    sunday = datetime(2026, 8, 23, 11, 30, tzinfo=TPE)
    night = datetime(2026, 8, 21, 20, 0, tzinfo=TPE)
    assert in_tw_session(lunch)
    assert not in_tw_session(sunday)
    assert not in_tw_session(night)


def test_write_html(tmp_path: Path | None = None) -> None:
    df = make_v_bounce_bars()
    sigs = detect_signals(df)
    trades = simulate(df, sigs)
    out_dir = tmp_path or Path("/tmp/tw5m_bounce_test")
    html = out_dir / "index.html"
    row = {"code": "6239", "name": "力成", "symbol": "6239.TW"}
    hits = [(row, sigs[0], trades[0], df)]
    path = write_html_report(html, hits, [row], "5d")
    text = path.read_text(encoding="utf-8")
    assert "破底反彈" in text
    assert "6239" in text
    assert "間隔" in text
    assert (path.parent / "img").exists()


def main() -> int:
    test_sma()
    test_parse_symbols()
    test_merge_universe_and_day_filter()
    test_detect_v_bounce_like_6239()
    test_skip_before_filters_open()
    test_flat_market_has_no_signal()
    test_tangled_ribbon_skipped()
    test_stack_pretty_rejects_glued_mas()
    test_drop_incomplete_5m()
    test_shallow_dip_ignored()
    test_simulate_and_summarize()
    test_signal_key_and_alert_text()
    test_in_tw_session()
    test_write_html()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

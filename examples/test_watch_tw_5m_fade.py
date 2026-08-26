#!/usr/bin/env python3
"""Synthetic tests for TW 5m 衝高回落 → 5/10/20 空頭排列 (no network)."""

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
    peak_clear_of_mas,
    signal_key,
    simulate,
    sma,
    stack_pretty,
    summarize_trades,
    write_html_report,
)


def _session_index(n: int, start: str = "2026-08-21 09:00") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="5min", tz=TPE)


def _ohlc(close: np.ndarray, spike_i: int, peak: float) -> pd.DataFrame:
    high = close + 0.35
    low = close - 0.35
    high[spike_i] = peak
    low[spike_i] = min(close[spike_i], peak) - 0.2
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


def make_v_fade_bars(n: int = 90) -> pd.DataFrame:
    """谷底 100 → 急拉 113 → 回落站下均線，讓 5/10/20 重新空頭排列。"""
    close = np.full(n, 100.0, dtype=float)
    for i in range(1, 52):
        close[i] = 100.4 - (i % 5) * 0.15
    spike_i = 58
    close[52:spike_i] = np.linspace(100.2, 106.0, spike_i - 52, endpoint=False)
    close[spike_i] = 111.0
    close[spike_i + 1] = 108.5
    close[spike_i + 2] = 106.0
    close[spike_i + 3] = 103.8
    close[spike_i + 4] = 102.0
    close[spike_i + 5] = 100.6
    close[spike_i + 6] = 99.4
    close[spike_i + 7] = 98.6
    for i in range(spike_i + 8, n):
        close[i] = 98.8 - (i - spike_i - 8) * 0.12
    df = _ohlc(close, spike_i, 113.0)
    df.loc[df.index[spike_i], "Volume"] = 4800.0
    df.loc[df.index[spike_i + 6], "Volume"] = 3600.0
    return df


def make_tangled_fade_bars(n: int = 90) -> pd.DataFrame:
    """急拉後在均線附近橫盤磨，5/10/20 黏在一起。"""
    close = np.full(n, 212.0, dtype=float)
    for i in range(1, 52):
        close[i] = 211.8 + (i % 3) * 0.04
    spike_i = 58
    close[52:spike_i] = np.linspace(211.8, 214.2, spike_i - 52, endpoint=False)
    close[spike_i] = 215.0
    wobble = np.array([0.10, -0.08, 0.06, -0.05, 0.08, -0.07, 0.05, -0.04, 0.07, -0.06, 0.04, -0.03])
    for j, i in enumerate(range(spike_i + 1, n)):
        close[i] = 212.85 + wobble[j % len(wobble)]
    return _ohlc(close, spike_i, 215.2)


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
    rows = parse_symbols("2303, 2330.TW, 6488.TWO")
    assert [r["symbol"] for r in rows] == ["2303.TW", "2330.TW", "6488.TWO"]
    assert rows[0]["code"] == "2303"
    assert rows[2]["market"] == "otc"


def test_merge_universe_and_day_filter() -> None:
    base = parse_symbols("2330")
    extra = parse_symbols("2303,2330")
    merged = merge_universe(base, extra)
    assert [r["code"] for r in merged] == ["2330", "2303"]
    df = make_v_fade_bars()
    sig = detect_signals(df)[0]
    assert hit_on_day(df, sig, df.index[sig.entry_idx].date())
    assert not hit_on_day(df, sig, datetime(2026, 1, 1).date())
    row = {"code": "2303", "close": 55.0}
    assert hit_within_max_price(row, sig, df, 400.0)
    assert not hit_within_max_price(row, sig, df, 50.0)
    expensive = {"code": "2327", "close": 590.0}
    assert not hit_within_max_price(expensive, sig, df, 400.0)
    prev = df.iloc[[0]].copy()
    prev.index = prev.index - pd.Timedelta(days=1)
    prev["High"] = 900.0
    week = pd.concat([prev, df])
    assert hit_within_max_price(row, detect_signals(week)[0], week, 400.0)


def test_detect_v_fade() -> None:
    df = make_v_fade_bars()
    sigs = detect_signals(df)
    assert sigs, "急拉 100→113 再回落後應出現 5/10/20 空頭排列"
    sig = sigs[0]
    assert sig.entry_idx > sig.break_idx
    assert sig.break_high >= 112.99
    assert sig.rally_pct >= 0.02
    assert sig.fade_pct >= 0.01
    assert sig.ma5 < sig.ma10 < sig.ma20
    assert sig.entry_idx - sig.break_idx <= 24


def test_skip_before_filters_open() -> None:
    df = make_v_fade_bars()
    assert detect_signals(df)
    assert detect_signals(df, skip_before=(23, 59)) == []


def test_flat_market_has_no_signal() -> None:
    assert detect_signals(make_flat_bars()) == []


def test_tangled_ribbon_skipped() -> None:
    df = make_tangled_fade_bars()
    pretty = detect_signals(df, require_pretty=True)
    loose = detect_signals(df, require_pretty=False)
    assert pretty == []
    assert len(pretty) <= len(loose)


def test_stack_pretty_rejects_glued_mas() -> None:
    n = 30
    ma5 = np.linspace(100.08, 100.02, n)
    ma10 = ma5 + 0.01
    ma20 = ma10 + 0.01
    close = ma5 - 0.02
    assert not stack_pretty(ma5, ma10, ma20, close, n - 1)
    ma5c = np.linspace(101.2, 100.0, n)
    ma10c = np.linspace(100.05, 99.95, n)
    ma20c = ma10c + 0.02
    closec = ma5c - 0.3
    assert not stack_pretty(ma5c, ma10c, ma20c, closec, n - 1)
    ma5b = np.array([100.0 - i * 0.35 for i in range(n)])
    ma10b = np.array([99.6 - i * 0.22 for i in range(n)])
    ma20b = np.array([99.3 - i * 0.12 for i in range(n)])
    # wait, for bearish we need ma5 < ma10 < ma20, so ma5 should be lowest
    ma5b = np.array([100.0 - i * 0.35 for i in range(n)])
    ma10b = np.array([100.4 - i * 0.22 for i in range(n)])
    ma20b = np.array([100.7 - i * 0.12 for i in range(n)])
    closeb = ma5b - 0.4
    assert stack_pretty(ma5b, ma10b, ma20b, closeb, n - 1)


def test_peak_clear_of_mas() -> None:
    assert peak_clear_of_mas(113.0, 110.8, 110.0, 109.5, 104.6, 104.0, 106.0)
    assert peak_clear_of_mas(113.0, 110.8, np.nan, np.nan)
    assert not peak_clear_of_mas(95.5, 94.5, 93.4, 92.3, 92.2, 95.8, 98.2)
    df = make_v_fade_bars()
    assert detect_signals(df)


def make_fade_capped_by_long_ma(n: int = 280) -> pd.DataFrame:
    """急拉仍停在長均下方，衝高那根上面還壓著 60/120/240。"""
    close = 120.0 - np.linspace(0, 30, n)
    spike_i = n - 20
    close[spike_i] = close[spike_i - 1] * 1.03
    close[spike_i + 1 : spike_i + 8] = np.linspace(close[spike_i] - 1.2, close[spike_i - 1] - 0.8, 7)
    for i in range(spike_i + 8, n):
        close[i] = close[i - 1] - 0.15
    return _ohlc(close, spike_i, float(close[spike_i]) + 0.4)


def test_spike_with_ma_overhead_skipped() -> None:
    n = 280
    df = make_fade_capped_by_long_ma(n)
    close = df["Close"].to_numpy(float)
    high = df["High"].to_numpy(float)
    spike_i = n - 20
    vals = [float(sma(close, p)[spike_i]) for p in (5, 10, 20, 60, 120, 240)]
    assert vals[-1] == vals[-1]
    assert vals[-1] > float(high[spike_i])
    assert not peak_clear_of_mas(float(high[spike_i]), *vals)


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


def test_shallow_rally_ignored() -> None:
    close = np.full(80, 100.0)
    close[50] = 100.6  # 0.6%，低於 2% 門檻
    df = _ohlc(close, 50, 100.7)
    assert detect_signals(df) == []


def test_simulate_short_and_summarize() -> None:
    df = make_v_fade_bars()
    sigs = detect_signals(df)
    trades = simulate(df, sigs)
    assert trades
    stats = summarize_trades(trades)
    assert stats["count"] == len(trades)
    t = trades[0]
    assert t.exit_idx >= t.entry_idx
    assert t.stop_price > t.entry_price
    assert t.target_price < t.entry_price
    assert t.exit_reason in {"stop", "target", "eod"}


def test_signal_key_and_alert_text() -> None:
    df = make_v_fade_bars()
    sig = detect_signals(df)[0]
    row = {"code": "2303", "name": "聯電", "symbol": "2303.TW"}
    key = signal_key(row, df, sig)
    assert key.startswith("2303.TW|")
    text = fmt_alert(row, df, sig)
    assert "2303" in text
    assert "空頭排列" in text
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
    df = make_v_fade_bars()
    sigs = detect_signals(df)
    trades = simulate(df, sigs)
    out_dir = tmp_path or Path("/tmp/tw5m_fade_test")
    html = out_dir / "index.html"
    row = {"code": "2303", "name": "聯電", "symbol": "2303.TW"}
    hits = [(row, sigs[0], trades[0], df)]
    path = write_html_report(html, hits, [row], "5d")
    text = path.read_text(encoding="utf-8")
    assert "衝高回落" in text
    assert "2303" in text
    assert "間隔" in text
    assert "做空" in text
    assert (path.parent / "img").exists()


def main() -> int:
    test_sma()
    test_parse_symbols()
    test_merge_universe_and_day_filter()
    test_detect_v_fade()
    test_skip_before_filters_open()
    test_flat_market_has_no_signal()
    test_tangled_ribbon_skipped()
    test_stack_pretty_rejects_glued_mas()
    test_peak_clear_of_mas()
    test_spike_with_ma_overhead_skipped()
    test_drop_incomplete_5m()
    test_shallow_rally_ignored()
    test_simulate_short_and_summarize()
    test_signal_key_and_alert_text()
    test_in_tw_session()
    test_write_html()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

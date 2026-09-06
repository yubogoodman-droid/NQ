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
    bounce_volume_ratio,
    climax_volume_ratio,
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
    trough_clear_of_mas,
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
    # 5 拉開了但 10/20 還黏著，一樣不算漂亮
    ma5c = np.linspace(100.0, 101.2, n)
    ma10c = np.linspace(99.95, 100.05, n)
    ma20c = ma10c - 0.02
    closec = ma5c + 0.3
    assert not stack_pretty(ma5c, ma10c, ma20c, closec, n - 1)
    ma5b = np.array([100.0 + i * 0.35 for i in range(n)])
    ma10b = np.array([99.6 + i * 0.22 for i in range(n)])
    ma20b = np.array([99.3 + i * 0.12 for i in range(n)])
    closeb = ma5b + 0.4
    assert stack_pretty(ma5b, ma10b, ma20b, closeb, n - 1)


def test_trough_clear_of_mas() -> None:
    assert trough_clear_of_mas(259.0, 260.8, 261.0, 261.5, 267.6, 268.0, 266.0)
    assert trough_clear_of_mas(259.0, 260.8, np.nan, np.nan)  # 長均還沒畫不算
    assert not trough_clear_of_mas(95.5, 96.5, 97.4, 98.3, 98.2, 95.2, 91.2)
    df = make_v_bounce_bars()
    assert detect_signals(df)


def make_bounce_resting_on_long_ma(n: int = 280) -> pd.DataFrame:
    """急殺仍停在長均上方，破底那根底下還墊著 60/120/200/240。"""
    close = 90.0 + np.linspace(0, 30, n)
    dump_i = n - 20
    close[dump_i] = close[dump_i - 1] * 0.97
    close[dump_i + 1 : dump_i + 8] = np.linspace(close[dump_i] + 1.2, close[dump_i - 1] + 0.8, 7)
    for i in range(dump_i + 8, n):
        close[i] = close[i - 1] + 0.15
    return _ohlc(close, dump_i, float(close[dump_i]) - 0.4)


def test_dump_with_ma_underneath_skipped() -> None:
    n = 280
    df = make_bounce_resting_on_long_ma(n)
    close = df["Close"].to_numpy(float)
    low = df["Low"].to_numpy(float)
    dump_i = n - 20
    vals = [float(sma(close, p)[dump_i]) for p in (5, 10, 20, 60, 120, 200, 240)]
    assert vals[-1] == vals[-1]  # not NaN
    assert vals[-1] < float(low[dump_i])
    assert not trough_clear_of_mas(float(low[dump_i]), *vals)


def test_volume_ratios() -> None:
    vol = np.full(60, 1000.0)
    vol[40] = 5500.0  # 破底那根爆量
    vol[41:50] = 1800.0  # 反彈帶量
    assert abs(climax_volume_ratio(vol, 40) - 5.5) < 1e-9
    assert abs(bounce_volume_ratio(vol, 40, 49) - 1.8) < 1e-9
    quiet = np.full(60, 1000.0)
    assert abs(climax_volume_ratio(quiet, 40) - 1.0) < 1e-9
    assert np.isnan(climax_volume_ratio(np.zeros(60), 40))


def test_dry_bounce_without_volume_skipped() -> None:
    """同一張 V 彈圖，把量抹平：富喬標準下不算，關掉量的門檻才算。"""
    df = make_v_bounce_bars()
    strict = detect_signals(df)
    assert strict
    sig = strict[0]
    assert sig.climax_ratio >= 2.0
    assert sig.bounce_vol_ratio >= 1.0
    assert sig.entry_price > sig.ma60
    flat = df.copy()
    flat["Volume"] = 1200.0
    assert detect_signals(flat) == []
    loose = detect_signals(flat, min_climax_vol=0, min_bounce_vol=0)
    assert loose and loose[0].entry_idx == sig.entry_idx
    # 只有破底爆量、反彈縮量也不算
    dump_only = flat.copy()
    dump_only.loc[dump_only.index[sig.break_idx], "Volume"] = 6000.0
    dump_only.loc[dump_only.index[sig.break_idx + 1 :], "Volume"] = 700.0
    assert detect_signals(dump_only) == []
    assert detect_signals(dump_only, min_bounce_vol=0)


def test_entry_below_ma60_skipped() -> None:
    df = make_v_bounce_bars()
    sig = detect_signals(df)[0]
    lifted = df.copy()
    # 把前面的高原抬高，讓 60MA 壓在進場價之上
    head = lifted.index[: sig.break_idx - 6]
    for col in ("Open", "High", "Low", "Close"):
        lifted.loc[head, col] = lifted.loc[head, col] + 6.0
    ma60 = sma(lifted["Close"].to_numpy(float), 60)
    assert float(lifted["Close"].iloc[sig.entry_idx]) < ma60[sig.entry_idx]
    assert all(s.entry_price > s.ma60 for s in detect_signals(lifted))
    assert len(detect_signals(lifted, require_above_ma60=False)) >= len(detect_signals(lifted))


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
    assert "MA60" in text
    assert "MA200" in text
    assert "破底量" in text
    assert "反彈量" in text
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
    test_trough_clear_of_mas()
    test_dump_with_ma_underneath_skipped()
    test_volume_ratios()
    test_dry_bounce_without_volume_skipped()
    test_entry_below_ma60_skipped()
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

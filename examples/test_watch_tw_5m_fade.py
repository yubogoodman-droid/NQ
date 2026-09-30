#!/usr/bin/env python3
"""Synthetic tests for TW 5m 國巨-style 空排跌破 MA240 (no network)."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scan_tw_ma_reclaim import TPE  # noqa: E402
from watch_tw_5m_fade import (  # noqa: E402
    MIN_BREAK_PCT,
    detect_signals,
    drop_incomplete_5m,
    fmt_alert,
    hit_on_day,
    hit_within_max_price,
    in_tw_session,
    merge_universe,
    parse_symbols,
    prior_session_min_over,
    ribbon_down,
    signal_key,
    simulate,
    sma,
    summarize_trades,
    write_html_report,
)

BARS_PER_DAY = 54  # 09:00–13:25


def _session_index(n: int, start: str = "2026-08-17 09:00") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="5min", tz=TPE)


def _tw_session_index(n_days: int, start: str = "2026-08-17") -> pd.DatetimeIndex:
    d = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=TPE)
    stamps: list = []
    while len(stamps) < n_days * BARS_PER_DAY:
        if d.weekday() < 5:
            day0 = d.replace(hour=9, minute=0, second=0, microsecond=0)
            stamps.extend(day0 + pd.Timedelta(minutes=5 * k) for k in range(BARS_PER_DAY))
        d += pd.Timedelta(days=1)
    return pd.DatetimeIndex(stamps[: n_days * BARS_PER_DAY])


def _ohlc_from_close(close: np.ndarray, index: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    high = close + 0.25
    low = close - 0.25
    idx = index if index is not None else _session_index(len(close))
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


def make_ma240_break_bars(*, n_days: int = 7, dump_offset: int = 0) -> pd.DataFrame:
    """前幾日緩升站在年線上，最後一盤 dump_offset 根後急殺跌破 MA240。"""
    idx = _tw_session_index(n_days)
    n = len(idx)
    dump_i = (n_days - 1) * BARS_PER_DAY + dump_offset
    climb = np.linspace(90.0, 118.0, dump_i)
    rest = np.full(n - dump_i, 100.0)
    close = np.r_[climb, rest]
    df = _ohlc_from_close(close, idx)
    df.iloc[dump_i, df.columns.get_loc("Volume")] = 4800.0
    return df


def make_flat_above_ma240(n: int = 260) -> pd.DataFrame:
    close = 100.0 + np.linspace(0, 1.2, n)
    return _ohlc_from_close(close)


def make_hairline_ma240_nick() -> pd.DataFrame:
    """華新科／順達：貼年線刺一下，距年線約 0.12%。"""
    df = make_ma240_break_bars()
    dump_i = 6 * BARS_PER_DAY
    close = df["Close"].to_numpy(float).copy()
    ma240 = sma(close, 240)
    nick = float(ma240[dump_i]) * (1.0 - 0.0012)
    close[dump_i:] = nick
    return _ohlc_from_close(close, df.index)


def make_prior_session_already_broken() -> pd.DataFrame:
    """前一盤已經貼破年線，隔日再真下穿也不算國巨。"""
    n_days = 7
    idx = _tw_session_index(n_days)
    n = len(idx)
    prev_open = 5 * BARS_PER_DAY
    last_open = 6 * BARS_PER_DAY
    climb = np.linspace(90.0, 118.0, prev_open)
    probe = np.r_[climb, np.full(n - prev_open, climb[-1])]
    ma_est = sma(probe, 240)
    nick = float(ma_est[prev_open]) * (1.0 - 0.0015)
    close = np.r_[climb, np.full(n - prev_open, nick)]
    ma240 = sma(close, 240)
    close[last_open] = float(ma240[last_open]) * 1.008
    close[last_open + 1 :] = float(ma240[last_open]) * 0.985
    return _ohlc_from_close(close, idx)


def make_same_day_recross() -> pd.DataFrame:
    """同一天跌破、很快彈回年線上、再跌破。"""
    df = make_ma240_break_bars()
    dump_i = 6 * BARS_PER_DAY
    close = df["Close"].to_numpy(float).copy()
    ma240 = sma(close, 240)
    k_up, k_dn = dump_i + 2, dump_i + 3
    close[k_up] = float(ma240[k_up]) * 1.012
    close[k_dn:] = 100.0
    return _ohlc_from_close(close, df.index)


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
    assert sigs, "箱體站上後急殺跌破 MA240 應出訊號"
    sig = sigs[0]
    assert sig.entry_idx == sig.break_idx
    assert sig.entry_price < sig.ma240
    assert sig.dist_pct >= MIN_BREAK_PCT
    assert sig.prior_over >= 0.005
    ma240s = sma(df["Close"].to_numpy(float), 240)
    assert sig.prev_close >= float(ma240s[sig.entry_idx - 1])
    assert sig.ma5 < sig.ma10 < sig.ma20
    assert sig.entry_price < sig.ma5
    ts = df.index[sig.entry_idx]
    assert (ts.hour, ts.minute) == (9, 0)


def test_yangming_like_open_dump_not_skipped() -> None:
    """開盤後不久跌破也要抓（陽明 09:05），前一盤箱體站上，不要擋 09:30 前。"""
    df = make_ma240_break_bars(dump_offset=1)
    sigs = detect_signals(df)
    assert sigs
    ts = df.index[sigs[0].entry_idx]
    assert (ts.hour, ts.minute) == (9, 5)
    early = detect_signals(df, skip_before=(9, 30))
    late = detect_signals(df, skip_before=None)
    assert len(late) >= len(early)


def test_hairline_nick_rejected() -> None:
    df = make_hairline_ma240_nick()
    assert detect_signals(df) == []
    loose = detect_signals(df, min_break_pct=0.0, min_prior_over_pct=0.0, one_per_day=False)
    assert loose, "關掉國巨濾網時貼線刺一下仍會出"


def test_prior_session_already_broken_rejected() -> None:
    df = make_prior_session_already_broken()
    assert detect_signals(df) == []


def test_one_signal_per_day() -> None:
    df = make_same_day_recross()
    once = detect_signals(df)
    many = detect_signals(df, one_per_day=False)
    assert once
    assert len(once) == 1
    assert len(many) >= 2
    assert {df.index[s.entry_idx].date() for s in many} == {df.index[once[0].entry_idx].date()}


def test_prior_session_ignores_today() -> None:
    """陽明 09:05：前一根 09:00 可能只貼 0.5%，要用前一交易日。"""
    df = make_ma240_break_bars(dump_offset=1)
    sigs = detect_signals(df)
    assert sigs
    i = sigs[0].entry_idx
    close = df["Close"].to_numpy(float)
    ma240 = sma(close, 240)
    prior = prior_session_min_over(df.index, close, ma240, i)
    assert prior is not None
    assert prior >= 0.005
    assert df.index[i - 1].date() == df.index[i].date()


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
    assert "前一盤" in text


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
    row = {"code": "2327", "name": "國巨", "symbol": "2327.TW"}
    hits = [(row, sigs[0], trades[0], df)]
    path = write_html_report(html, hits, [row], "7d · 國巨濾網")
    text = path.read_text(encoding="utf-8")
    assert "MA240" in text
    assert "2327" in text
    assert "空頭排列" in text
    assert "國巨" in text
    assert (path.parent / "img").exists()


def main() -> int:
    test_sma()
    test_parse_symbols()
    test_ribbon_down()
    test_detect_ma240_break()
    test_yangming_like_open_dump_not_skipped()
    test_hairline_nick_rejected()
    test_prior_session_already_broken_rejected()
    test_one_signal_per_day()
    test_prior_session_ignores_today()
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

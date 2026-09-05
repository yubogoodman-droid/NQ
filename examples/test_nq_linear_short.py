#!/usr/bin/env python3
"""Synthetic tests for NQ 1m 線性空 (no Yahoo)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nq_linear_short import (  # noqa: E402
    ET,
    hour_bar_index,
    hour_window,
    long_lower_wick,
    map_closed_1h,
    normalize_1h_index,
    parse_period_days,
    resample_5m,
    resample_1h,
    run_linear_short,
    sma,
    summarize_trades,
    write_html_report,
)


def test_parse_period_days() -> None:
    assert parse_period_days("7d") == 7
    assert parse_period_days("8d") == 8
    assert parse_period_days("1w") == 7
    assert parse_period_days("1mo") == 30


def test_sma() -> None:
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = sma(arr, 3)
    assert np.isnan(out[1])
    assert abs(out[2] - 2.0) < 1e-9
    assert abs(out[4] - 4.0) < 1e-9


def test_resample_and_map_1h() -> None:
    idx = pd.date_range("2026-08-24 09:00", periods=180, freq="1min", tz=ET)
    close = 20000.0 + np.arange(180, dtype=float)
    df = pd.DataFrame(
        {
            "Open": close,
            "High": close + 0.5,
            "Low": close - 0.5,
            "Close": close,
            "Volume": 1.0,
        },
        index=idx,
    )
    h1 = resample_1h(df)
    assert len(h1) >= 2
    # 09:00 那根小時要等 10:00 才算收盤
    mapped = map_closed_1h(df, h1, "Close")
    at_0959 = int(np.where(df.index == pd.Timestamp("2026-08-24 09:59", tz=ET))[0][0])
    at_1000 = int(np.where(df.index == pd.Timestamp("2026-08-24 10:00", tz=ET))[0][0])
    assert np.isnan(mapped[at_0959])
    assert not np.isnan(mapped[at_1000])
    assert abs(mapped[at_1000] - float(h1.iloc[0]["Close"])) < 1e-9
    # 10:00 對到的是 09:00-10:00 收盤，不能是還在走的 10 點那根
    assert mapped[at_1000] < float(df["Close"].iloc[at_1000])
    m5 = resample_5m(df)
    assert len(m5) == 36
    assert abs(float(m5.iloc[0]["Open"]) - 20000.0) < 1e-9
    assert abs(float(m5.iloc[0]["Close"]) - 20004.0) < 1e-9


def test_hour_window_keeps_full_bars() -> None:
    idx = pd.date_range("2026-08-24 00:00", periods=48, freq="1h", tz=ET)
    df = pd.DataFrame(
        {
            "Open": 20000.0,
            "High": 20010.0,
            "Low": 19990.0,
            "Close": 20005.0,
            "Volume": 10.0,
        },
        index=idx,
    )
    entry = pd.Timestamp("2026-08-24 11:08", tz=ET)
    exit_ = pd.Timestamp("2026-08-24 11:40", tz=ET)
    w = hour_window(df, entry, exit_, before=8, after=2)
    assert w.index[0] == pd.Timestamp("2026-08-24 03:00", tz=ET)
    assert w.index[-1] == pd.Timestamp("2026-08-24 13:00", tz=ET)
    assert all(ts.minute == 0 and ts.second == 0 for ts in w.index)
    assert hour_bar_index(w, entry) == list(w.index).index(pd.Timestamp("2026-08-24 11:00", tz=ET))
    assert hour_bar_index(w, exit_) == hour_bar_index(w, entry)
    messy = df.copy()
    messy.index = messy.index + pd.Timedelta(minutes=17)
    w2 = hour_window(normalize_1h_index(messy), entry, exit_, before=2, after=1)
    assert all(ts.minute == 0 for ts in w2.index)


def test_long_lower_wick() -> None:
    assert long_lower_wick(100.0, 101.0, 90.0, 99.5) is True
    assert long_lower_wick(100.0, 101.0, 99.6, 100.2) is False


def test_summarize_trades() -> None:
    class T:
        def __init__(self, pnl: float, reason: str = "ma200"):
            self.pnl_points = pnl
            self.exit_reason = reason

    stats = summarize_trades([T(12.0, "ma200"), T(-4.0, "stop"), T(3.0, "wick")])
    assert stats["count"] == 3
    assert stats["wins"] == 2
    assert abs(stats["total_points"] - 11.0) < 1e-9
    assert stats["by_reason"]["ma200"]["n"] == 1


def _df_from_ohlc(o, h, l, c) -> pd.DataFrame:
    n = len(c)
    idx = pd.date_range("2026-08-24 09:30", periods=n, freq="1min", tz=ET)
    return pd.DataFrame(
        {
            "Open": np.asarray(o, float),
            "High": np.asarray(h, float),
            "Low": np.asarray(l, float),
            "Close": np.asarray(c, float),
            "Volume": np.full(n, 80.0),
        },
        index=idx,
    )


def _uptrend(n: int = 280, start: float = 20000.0, step: float = 1.6, wick: float = 0.4):
    close = start + np.arange(n, dtype=float) * step
    high = close + wick
    low = close - wick
    open_ = np.r_[close[0], close[:-1]]
    return open_, high, low, close


def _append(o, h, l, c, oo, hh, ll, cc):
    return (
        np.r_[o, oo],
        np.r_[h, hh],
        np.r_[l, ll],
        np.r_[c, cc],
    )


def _make_valid_setup(*, after: str = "ma200") -> pd.DataFrame:
    """Uptrend 4H high → MA10 retest → MA20 break, then an exit path."""
    o, h, l, c = _uptrend(280, step=1.6)
    last = float(c[-1])
    # 回測 MA10：低點碰到均線附近，收盤仍在 MA20 上
    o, h, l, c = _append(o, h, l, c, last, last, last - 10.0, last - 4.0)
    mid = float(c[-1])
    o, h, l, c = _append(o, h, l, c, mid, mid + 0.3, mid - 8.0, mid - 6.0)
    # 明確跌破 MA20
    br = float(c[-1])
    o, h, l, c = _append(o, h, l, c, br, br + 0.2, br - 14.0, br - 12.0)

    if after == "ma200":
        px = float(c[-1])
        for _ in range(40):
            nxt = px - 12.0
            o, h, l, c = _append(o, h, l, c, px, px + 0.2, nxt - 0.4, nxt)
            px = nxt
    elif after == "stop":
        px = float(c[-1])
        peak = float(np.max(h))
        o, h, l, c = _append(o, h, l, c, px, peak + 2.0, px - 1.0, px + 1.0)
    elif after == "wick":
        px = float(c[-1])
        for _ in range(10):
            nxt = px - 6.0
            o, h, l, c = _append(o, h, l, c, px, px + 0.2, nxt, nxt)
            px = nxt
        # 靠近 MA200 的長下影（低點不跌破 MA200）
        ma200 = float(np.mean(c[-200:]))
        low = ma200 + 3.0
        o, h, l, c = _append(o, h, l, c, px, px + 0.4, low, px - 0.8)
    elif after == "new_high":
        px = float(c[-2])  # before the MA20 break bar? after retest, before break
        # Rebuild: uptrend + retest only, then new high
        o, h, l, c = _uptrend(280, step=1.6)
        last = float(c[-1])
        o, h, l, c = _append(o, h, l, c, last, last, last - 10.0, last - 4.0)
        peak = float(np.max(h))
        o, h, l, c = _append(o, h, l, c, last - 4.0, peak + 3.0, last - 5.0, last - 3.0)
    elif after == "flat":
        px = float(c[-1])
        for _ in range(8):
            o, h, l, c = _append(o, h, l, c, px, px + 0.3, px - 0.3, px)

    return _df_from_ohlc(o, h, l, c)


def test_entry_and_ma200_tp() -> None:
    df = _make_valid_setup(after="ma200")
    trades = run_linear_short(df)
    assert trades, "expected a linear-short after 4H high / MA10 retest / MA20 break"
    t = trades[0]
    assert t.signal.peak_dist >= 100
    assert t.entry_price < t.stop_price
    assert t.exit_reason == "ma200"
    assert t.pnl_points > 0
    assert t.exit_idx > t.entry_idx


def test_stop_exit() -> None:
    df = _make_valid_setup(after="stop")
    trades = run_linear_short(df)
    assert trades
    assert trades[0].exit_reason == "stop"
    assert trades[0].pnl_points < 0
    assert abs(trades[0].exit_price - trades[0].stop_price) < 1e-9


def test_wick_exit() -> None:
    df = _make_valid_setup(after="wick")
    trades = run_linear_short(df)
    assert trades
    assert trades[0].exit_reason == "wick"
    assert trades[0].pnl_points > 0


def test_skip_small_peak_dist() -> None:
    o, h, l, c = _uptrend(280, step=0.25)
    last = float(c[-1])
    o, h, l, c = _append(o, h, l, c, last, last, last - 4.0, last - 1.5)
    mid = float(c[-1])
    o, h, l, c = _append(o, h, l, c, mid, mid + 0.2, mid - 6.0, mid - 5.0)
    df = _df_from_ohlc(o, h, l, c)
    funnel: dict = {}
    trades = run_linear_short(df, funnel=funnel)
    assert not trades
    assert funnel.get("skip_dist", 0) >= 1 or funnel.get("taken", 0) == 0


def test_cancel_on_new_high() -> None:
    df = _make_valid_setup(after="new_high")
    funnel: dict = {}
    trades = run_linear_short(df, funnel=funnel)
    assert not trades
    assert funnel.get("cancel_new_high", 0) >= 1
    assert funnel.get("taken", 0) == 0


def test_no_same_bar_exit() -> None:
    """進場當根即使低點已到 MA200 也不出場（對齊 Pine position_size）。"""
    df = _make_valid_setup(after="flat")
    # 把進場那根改成一路殺到 MA200
    trades0 = run_linear_short(df)
    assert trades0
    entry = trades0[0].entry_idx
    df = df.copy()
    ma200 = float(df["Close"].rolling(200).mean().iloc[entry])
    df.iat[entry, df.columns.get_loc("Low")] = ma200 - 5.0
    df.iat[entry, df.columns.get_loc("Close")] = df["Close"].iloc[entry]
    trades = run_linear_short(df)
    assert trades
    assert trades[0].entry_idx == entry
    assert trades[0].exit_idx > entry or trades[0].exit_reason == "eod"


def test_write_html_report() -> None:
    df = _make_valid_setup(after="ma200")
    trades = run_linear_short(df)
    out = Path("/tmp/nq_linear_short_test.html")
    path = write_html_report(out, df, trades, "NQ=F", "demo")
    text = path.read_text(encoding="utf-8")
    assert "線性空" in text
    if trades:
        assert "<img src='img/" in text
        assert "1m" in text
        assert "1h" in text
        assert any((path.parent / "img").glob("t01_*.png"))


def main() -> int:
    test_parse_period_days()
    test_sma()
    test_resample_and_map_1h()
    test_hour_window_keeps_full_bars()
    test_long_lower_wick()
    test_summarize_trades()
    test_entry_and_ma200_tp()
    test_stop_exit()
    test_wick_exit()
    test_skip_small_peak_dist()
    test_cancel_on_new_high()
    test_no_same_bar_exit()
    test_write_html_report()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

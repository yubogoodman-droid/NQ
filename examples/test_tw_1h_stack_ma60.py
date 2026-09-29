#!/usr/bin/env python3
"""Synthetic tests for 台股 1h MA5>MA10>MA20 且站上 MA60（不打網路）。"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tw_1h_stack_ma60 import (  # noqa: E402
    TPE,
    bar_close_time,
    current_setup,
    detect_signals,
    drop_forming,
    filter_recent,
    hit_key,
    Hit,
    next_scan_time,
    setup_at,
    sma,
    Signal,
)

HOURS = (9, 10, 11, 12, 13)


def session_index(n: int, start: str = "2026-06-01 09:00") -> pd.DatetimeIndex:
    t0 = pd.Timestamp(start, tz=TPE)
    d = t0.date()
    hi = HOURS.index(t0.hour) if t0.hour in HOURS else 0
    times = []
    while len(times) < n:
        if d.weekday() < 5:
            times.append(datetime(d.year, d.month, d.day, HOURS[hi], 0, tzinfo=TPE))
            hi += 1
            if hi >= len(HOURS):
                hi = 0
                d += timedelta(days=1)
        else:
            d += timedelta(days=1)
            hi = 0
    return pd.DatetimeIndex(times)


def ohlc_from_close(closes) -> pd.DataFrame:
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    df = pd.DataFrame(
        {
            "Open": opens,
            "High": closes + 0.3,
            "Low": closes - 0.4,
            "Close": closes,
            "Volume": np.full(n, 1000.0),
        },
        index=session_index(n),
    )
    return df


def uptrend_closes(n: int = 80, start: float = 90.0, step: float = 0.25) -> np.ndarray:
    return start + step * np.arange(n, dtype=float)


def test_setup_needs_stack_and_ma60() -> None:
    close = np.array([10.0, 10.0, 10.0])
    ma5 = np.array([9.0, 9.0, 8.0])
    ma10 = np.array([8.0, 9.5, 7.5])
    ma20 = np.array([7.0, 8.0, 7.0])
    ma60 = np.array([9.5, 9.5, 11.0])
    assert setup_at(0, close, ma5, ma10, ma20, ma60)  # stacked and close > ma60
    assert not setup_at(1, close, ma5, ma10, ma20, ma60)  # 5 < 10
    assert not setup_at(2, close, ma5, ma10, ma20, ma60)  # close < ma60


def test_first_valid_ma60_bar_is_a_signal() -> None:
    df = ohlc_from_close(uptrend_closes(80))
    sigs = detect_signals(df)
    assert sigs
    first = sigs[0]
    assert first.idx == 59  # MA60 第一根可算的 bar（0-based）
    assert first.ma5 > first.ma10 > first.ma20
    assert first.close > first.ma60


def test_reclaim_ma60_while_stacked() -> None:
    # 長盤整後緩升，MA60 貼近現價，才能小跌破再站回且 5>10>20 還在。
    base = [100.0] * 50 + list(np.round(100.1 + 0.1 * np.arange(20), 10))
    m60 = float(sma(np.asarray(base, dtype=float), 60)[-1])
    df = ohlc_from_close(base + [m60 - 0.05, base[-1]])
    close = df["Close"].to_numpy(float)
    ma60 = sma(close, 60)
    assert close[-3] > ma60[-3]
    assert close[-2] < ma60[-2]
    assert close[-1] > ma60[-1]
    sigs = detect_signals(df)
    assert sigs[-1].idx == len(df) - 1
    assert sigs[-1].close > sigs[-1].ma60
    assert sigs[-1].ma5 > sigs[-1].ma10 > sigs[-1].ma20


def test_no_repeat_while_already_standing() -> None:
    df = ohlc_from_close(uptrend_closes(80))
    sigs = detect_signals(df)
    # 上升趨勢裡，一旦站上就持續滿足，不該每根都發
    assert len(sigs) == 1


def test_stack_forms_while_already_above_ma60() -> None:
    """價已在 MA60 上，短均交叉成 5>10>20 才通知。"""
    n = 80
    closes = np.full(n, 100.0)
    # 先讓短均糾結：後半段微幅向下再向上
    closes[:60] = 100 + 0.05 * np.arange(60)
    closes[60:70] = closes[59] - 0.4 * np.arange(1, 11)  # 短均翻空，但仍高於 MA60
    closes[70:] = closes[69] + 0.8 * np.arange(1, n - 69)  # 再拉，短均轉多
    df = ohlc_from_close(closes)
    sigs = detect_signals(df)
    assert sigs
    last = sigs[-1]
    assert last.idx >= 70
    assert last.ma5 > last.ma10 > last.ma20
    assert last.close > last.ma60
    # 轉多那根的前一根還沒排列
    close = df["Close"].to_numpy(float)
    ma5, ma10, ma20, ma60 = sma(close, 5), sma(close, 10), sma(close, 20), sma(close, 60)
    assert not setup_at(last.idx - 1, close, ma5, ma10, ma20, ma60)


def test_below_ma60_is_not_current() -> None:
    base = list(uptrend_closes(70))
    df = ohlc_from_close(base + [base[-1] - 30.0])
    assert current_setup(df) is None


def test_current_setup_when_standing() -> None:
    df = ohlc_from_close(uptrend_closes(80))
    sig = current_setup(df)
    assert sig is not None
    assert sig.idx == len(df) - 1
    assert sig.ma5 > sig.ma10 > sig.ma20
    assert sig.close > sig.ma60


def test_filter_recent_keeps_last_n() -> None:
    df = ohlc_from_close(uptrend_closes(80))
    fake = [
        Signal(idx=10, close=1, ma5=3, ma10=2, ma20=1, ma60=0.5),
        Signal(idx=78, close=1, ma5=3, ma10=2, ma20=1, ma60=0.5),
        Signal(idx=79, close=1, ma5=3, ma10=2, ma20=1, ma60=0.5),
    ]
    got = filter_recent(df, fake, 2)
    assert [s.idx for s in got] == [78, 79]


def test_drop_forming_hourly_and_1330() -> None:
    df = ohlc_from_close(uptrend_closes(10))
    last = df.index[-1]
    during = last + pd.Timedelta(minutes=20)
    closed = bar_close_time(last) + pd.Timedelta(minutes=1)
    assert len(drop_forming(df, during)) == len(df) - 1
    assert len(drop_forming(df, closed)) == len(df)

    ts = pd.Timestamp("2026-06-01 13:00", tz=TPE)
    assert bar_close_time(ts) == pd.Timestamp("2026-06-01 13:30", tz=TPE)
    ts2 = pd.Timestamp("2026-06-01 10:00", tz=TPE)
    assert bar_close_time(ts2) == pd.Timestamp("2026-06-01 11:00", tz=TPE)


def test_next_scan_skips_weekend() -> None:
    friday_after = datetime(2026, 6, 5, 14, 0, tzinfo=TPE)
    nxt = next_scan_time(friday_after)
    assert nxt.date() == datetime(2026, 6, 8).date()
    assert (nxt.hour, nxt.minute) == (10, 2)

    tue = datetime(2026, 6, 2, 9, 50, tzinfo=TPE)
    nxt2 = next_scan_time(tue)
    assert nxt2 == datetime(2026, 6, 2, 10, 2, tzinfo=TPE)


def test_hit_key_stable() -> None:
    df = ohlc_from_close(uptrend_closes(80))
    sig = detect_signals(df)[0]
    hit = Hit({"symbol": "2330.TW", "code": "2330", "name": "台積電"}, sig, df)
    k1 = hit_key(hit)
    k2 = hit_key(hit)
    assert k1 == k2
    assert k1.startswith("2330.TW|")


def main() -> int:
    tests = [
        test_setup_needs_stack_and_ma60,
        test_first_valid_ma60_bar_is_a_signal,
        test_reclaim_ma60_while_stacked,
        test_no_repeat_while_already_standing,
        test_stack_forms_while_already_above_ma60,
        test_below_ma60_is_not_current,
        test_current_setup_when_standing,
        test_filter_recent_keeps_last_n,
        test_drop_forming_hourly_and_1330,
        test_next_scan_skips_weekend,
        test_hit_key_stable,
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

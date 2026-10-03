#!/usr/bin/env python3
"""Synthetic tests for 台股 1h MA5>MA10>MA20 且站上 MA240（不打網路）。"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tw_1h_stack_ma240 import (  # noqa: E402
    Hit,
    Signal,
    TPE,
    above_ma240_at,
    bar_close_time,
    current_setup,
    detect_signals,
    drop_forming,
    ensure_watch,
    fill_fwd,
    filter_recent,
    hit_key,
    next_scan_time,
    rows_from_symbols,
    setup_at,
    sma,
    stacked_at,
    tw_tick,
)

HOURS = (9, 10, 11, 12, 13)


def session_index(n: int, start: str = "2026-01-05 09:00") -> pd.DatetimeIndex:
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


def uptrend_closes(n: int = 280, start: float = 90.0, step: float = 0.25) -> np.ndarray:
    return start + step * np.arange(n, dtype=float)


def test_tw_tick() -> None:
    assert tw_tick(9.9) == 0.01
    assert tw_tick(10) == 0.05
    assert tw_tick(65.1) == 0.10
    assert tw_tick(251.5) == 0.50
    assert tw_tick(689) == 1.00


def test_setup_needs_stack_and_ma240() -> None:
    close = np.array([10.0, 10.0, 10.0])
    ma5 = np.array([9.0, 9.0, 8.0])
    ma10 = np.array([8.0, 9.5, 7.5])
    ma20 = np.array([7.0, 8.0, 7.0])
    ma240 = np.array([9.5, 9.5, 11.0])
    assert setup_at(0, close, ma5, ma10, ma20, ma240)  # stacked and close 高於 MA240 至少 1 檔
    assert not setup_at(1, close, ma5, ma10, ma20, ma240)  # 5 < 10
    assert not setup_at(2, close, ma5, ma10, ma20, ma240)  # close < ma240


def test_first_valid_ma240_bar_is_a_signal() -> None:
    df = ohlc_from_close(uptrend_closes(280))
    sigs = detect_signals(df)
    assert sigs
    first = sigs[0]
    assert first.idx == 239  # MA240 第一根可算的 bar（0-based）
    assert first.ma5 > first.ma10 > first.ma20
    assert first.close > first.ma240


def _flat_then_reclaim(stand_extra: float) -> pd.DataFrame:
    # 長盤整後輕輕往上，MA240 貼近現價，才能跌破再站回且 5>10>20 還在。
    base = [100.0] * 240 + [100.02, 100.04, 100.06, 100.08, 100.10]
    m240 = float(sma(np.asarray(base, dtype=float), 240)[-1])
    return ohlc_from_close(base + [m240 - 0.01, m240 + stand_extra])


def test_reclaim_ma240_while_stacked() -> None:
    df = _flat_then_reclaim(0.60)  # 100 元檔 1 檔 = 0.5
    close = df["Close"].to_numpy(float)
    ma240 = sma(close, 240)
    assert close[-3] > ma240[-3]
    assert close[-2] < ma240[-2]
    assert close[-1] - ma240[-1] >= 0.50
    sigs = detect_signals(df)
    assert sigs[-1].idx == len(df) - 1
    assert not above_ma240_at(len(df) - 2, close, ma240)
    assert sigs[-1].close > sigs[-1].ma240
    assert sigs[-1].ma5 > sigs[-1].ma10 > sigs[-1].ma20


def test_kiss_ma240_less_than_one_tick_is_not_a_stand() -> None:
    """鴻海那種收 251.50 / MA240 251.49（不到 1 檔）不算站上。"""
    df = _flat_then_reclaim(0.01)  # << 0.5
    close = df["Close"].to_numpy(float)
    ma240 = sma(close, 240)
    assert close[-1] > ma240[-1]
    assert close[-1] - ma240[-1] < tw_tick(close[-1])
    sigs = detect_signals(df)
    assert all(s.idx != len(df) - 1 for s in sigs)
    assert current_setup(df) is None


def test_stack_forming_while_already_above_ma240_is_not_a_signal() -> None:
    """價已在 MA240 上，短均才交叉成 5>10>20 → 不算（要當下那根才站上）。"""
    n = 280
    closes = np.full(n, 100.0)
    closes[:240] = 100 + 0.04 * np.arange(240)
    closes[240:255] = closes[239] - 0.35 * np.arange(1, 16)
    closes[255:] = closes[254] + 0.7 * np.arange(1, n - 254)
    df = ohlc_from_close(closes)
    close = df["Close"].to_numpy(float)
    ma5, ma10, ma20, ma240 = sma(close, 5), sma(close, 10), sma(close, 20), sma(close, 240)
    stack_idxs = [
        i
        for i in range(1, n)
        if stacked_at(i, ma5, ma10, ma20)
        and not stacked_at(i - 1, ma5, ma10, ma20)
        and above_ma240_at(i, close, ma240)
        and above_ma240_at(i - 1, close, ma240)
    ]
    assert stack_idxs, "synthetic series never formed 5>10>20 while already above MA240"
    sigs = detect_signals(df)
    sig_idxs = {s.idx for s in sigs}
    assert stack_idxs[0] not in sig_idxs
    for sig in sigs:
        assert not above_ma240_at(sig.idx - 1, close, ma240)


def test_no_repeat_while_already_standing() -> None:
    df = ohlc_from_close(uptrend_closes(280))
    sigs = detect_signals(df)
    # 上升趨勢裡，一旦站上就持續滿足，不該每根都發
    assert len(sigs) == 1


def test_below_ma240_is_not_current() -> None:
    base = list(uptrend_closes(250))
    df = ohlc_from_close(base + [base[-1] - 40.0])
    assert current_setup(df) is None


def test_current_setup_requires_this_bar_stand() -> None:
    # 已經站在上面的最後一根不算
    df = ohlc_from_close(uptrend_closes(280))
    assert current_setup(df) is None
    # 這一根才從 MA240 下方站上（至少 1 檔）
    df2 = _flat_then_reclaim(0.60)
    sig = current_setup(df2)
    assert sig is not None
    assert sig.idx == len(df2) - 1
    assert sig.close - sig.ma240 >= tw_tick(sig.close) - 1e-9


def test_filter_recent_keeps_last_n() -> None:
    df = ohlc_from_close(uptrend_closes(280))
    fake = [
        Signal(idx=10, close=1, ma5=3, ma10=2, ma20=1, ma240=0.5),
        Signal(idx=278, close=1, ma5=3, ma10=2, ma20=1, ma240=0.5),
        Signal(idx=279, close=1, ma5=3, ma10=2, ma20=1, ma240=0.5),
    ]
    got = filter_recent(df, fake, 2)
    assert [s.idx for s in got] == [278, 279]


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
    df = ohlc_from_close(uptrend_closes(280))
    sig = detect_signals(df)[0]
    hit = Hit({"symbol": "3035.TW", "code": "3035", "name": "智原"}, sig, df)
    k1 = hit_key(hit)
    k2 = hit_key(hit)
    assert k1 == k2
    assert k1.startswith("3035.TW|")


def test_fill_fwd_uses_next_sessions() -> None:
    df = ohlc_from_close(uptrend_closes(280))
    sig = detect_signals(df)[0]
    hit = fill_fwd(Hit({"code": "3035", "name": "智原", "symbol": "3035.TW"}, sig, df))
    assert hit.fwd_1d is not None
    assert hit.fwd_3d is not None
    assert hit.fwd_1d > 0


def test_rows_from_symbols_and_watch() -> None:
    rows = rows_from_symbols("3035, 6488.TWO")
    assert [r["symbol"] for r in rows] == ["3035.TW", "6488.TWO"]
    assert rows[0]["name"] == "智原"
    uni = ensure_watch([{"code": "2330", "name": "台積電", "symbol": "2330.TW"}])
    assert uni[0]["code"] == "3035"
    assert "2330" in {r["code"] for r in uni}


def test_resolve_symbols_defaults_to_universe() -> None:
    from argparse import Namespace
    from tw_1h_stack_ma240 import resolve_symbols

    assert resolve_symbols(Namespace(symbols="", universe=False)) == ""
    assert resolve_symbols(Namespace(symbols="2330", universe=False)) == "2330"
    assert resolve_symbols(Namespace(symbols="2330", universe=True)) == ""


def main() -> int:
    tests = [
        test_tw_tick,
        test_setup_needs_stack_and_ma240,
        test_first_valid_ma240_bar_is_a_signal,
        test_reclaim_ma240_while_stacked,
        test_kiss_ma240_less_than_one_tick_is_not_a_stand,
        test_stack_forming_while_already_above_ma240_is_not_a_signal,
        test_no_repeat_while_already_standing,
        test_below_ma240_is_not_current,
        test_current_setup_requires_this_bar_stand,
        test_filter_recent_keeps_last_n,
        test_drop_forming_hourly_and_1330,
        test_next_scan_skips_weekend,
        test_hit_key_stable,
        test_fill_fwd_uses_next_sessions,
        test_rows_from_symbols_and_watch,
        test_resolve_symbols_defaults_to_universe,
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

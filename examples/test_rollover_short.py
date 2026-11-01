#!/usr/bin/env python3
"""合成 5 分 + 1 分 K 測試起跌放空。不打網路。

5 分走勢照 2026-10-31 SOXS 那一段：先墊高，極點 54.85，再以較低高點跌破頸線。
進場落在 1 分收盤。極點本身不是進場。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rollover.strategy import (  # noqa: E402
    RolloverParams,
    RolloverSignal,
    detect_rollover_shorts,
    inspect_rollover,
    is_us_cash_session,
    qty_for_risk,
    simulate_shorts,
)


def _frame(rows: list[tuple], start: str, freq: str) -> pd.DataFrame:
    index = pd.date_range(start, periods=len(rows), freq=freq, tz="UTC")
    return pd.DataFrame(rows, index=index, columns=["open", "high", "low", "close", "volume"])


def _expand_1m(df5: pd.DataFrame) -> pd.DataFrame:
    """每根 5 分拆成 5 根 1 分。第一根帶著這根 5 分的高低收，後面四根收在收盤價。"""

    records = []
    for ts, row in df5.iterrows():
        vol = float(row.volume) / 5.0
        close = float(row.close)
        records.append((ts, float(row.open), float(row.high), float(row.low), close, vol))
        for k in range(1, 5):
            t = ts + pd.Timedelta(minutes=k)
            records.append((t, close, close, close, close, vol))
    index = [item[0] for item in records]
    return pd.DataFrame(
        [(o, h, low, c, v) for _, o, h, low, c, v in records],
        index=pd.DatetimeIndex(index),
        columns=["open", "high", "low", "close", "volume"],
    )


def _pair(rows: list[tuple], start: str = "2026-10-31 00:00") -> tuple[pd.DataFrame, pd.DataFrame]:
    df5 = _frame(rows, start, "5min")
    return df5, _expand_1m(df5)


def _flat(n: int, price: float, volume: float = 1000.0) -> list[tuple]:
    return [(price, price + 0.10, price - 0.10, price, volume) for _ in range(n)]


def _soxs_top(break_volume: float = 4000.0, break_close: float = 52.05) -> list[tuple]:
    """40 根 5 分橫在 52，再衝 54.85，回檔頸線 52.35，較低高點 53.50，第 47 根嘗試跌破。"""

    rows = _flat(40, 52.0)
    rows.extend(
        [
            (53.5, 54.85, 53.40, 54.20, 3000),  # 40 極點
            (54.0, 54.00, 52.50, 52.70, 2000),  # 41
            (52.7, 52.80, 52.35, 52.60, 1500),  # 42 頸線
            (52.6, 53.20, 52.50, 53.00, 1200),  # 43
            (53.0, 53.50, 52.90, 53.30, 1200),  # 44 較低高點
            (53.3, 53.20, 52.80, 52.90, 1100),  # 45
            (52.9, 53.00, 52.70, 52.80, 1100),  # 46 較低高點在這根確認
            (52.8, 52.70, min(51.90, break_close), break_close, break_volume),  # 47
        ]
    )
    return rows


def _set_1m(df1: pd.DataFrame, i: int, *, high: float, low: float, close: float, volume: float) -> None:
    df1.iloc[i, df1.columns.get_loc("open")] = close
    df1.iloc[i, df1.columns.get_loc("high")] = high
    df1.iloc[i, df1.columns.get_loc("low")] = low
    df1.iloc[i, df1.columns.get_loc("close")] = close
    df1.iloc[i, df1.columns.get_loc("volume")] = volume


def test_us_cash_session() -> None:
    # 2026-10-28 是週三。07:15 UTC 還沒開盤；19:00 UTC = 美東 15:00，還在盤內。
    before = pd.Timestamp("2026-10-28 12:00", tz="UTC")
    inside = pd.Timestamp("2026-10-28 19:00", tz="UTC")
    after = pd.Timestamp("2026-10-30 21:00", tz="UTC")
    weekend = pd.Timestamp("2026-10-31 02:00", tz="UTC")
    assert not is_us_cash_session(before)
    assert is_us_cash_session(inside)
    assert not is_us_cash_session(after)
    assert not is_us_cash_session(weekend)


def test_qty_uses_stop_distance() -> None:
    assert qty_for_risk(52.0, 53.5, 150) == 100
    assert qty_for_risk(52.0, 53.5, 100) == 66
    assert qty_for_risk(53.0, 52.0, 100) == 0
    assert qty_for_risk(52.0, 52.0, 100) == 0


def test_uptrend_sell_marks_do_not_short() -> None:
    rows = []
    price = 49.0
    for i in range(70):
        price += 0.08
        volume = 5000.0 if i % 5 == 0 else 800.0
        close = price - 0.20 if i % 5 == 0 else price
        rows.append((price - 0.05, price + 0.12, price - 0.10, close, volume))
    df5, df1 = _pair(rows)
    assert detect_rollover_shorts(df5, df1) == []


def test_break_close_stops_at_lower_high_not_extreme() -> None:
    df5, df1 = _pair(_soxs_top())
    signals = detect_rollover_shorts(df5, df1)
    assert len(signals) == 1, signals
    sig = signals[0]
    assert sig.kind == "break"
    assert sig.bar_idx == 47 * 5
    assert sig.timestamp == df1.index[47 * 5]
    assert sig.entry == 52.05
    assert sig.neckline == 52.35
    assert sig.extreme == 54.85
    assert sig.lower_high == 53.50
    assert sig.lower_high_idx == 44
    assert sig.stop == 53.50
    assert sig.stop < sig.extreme
    assert sig.entry < sig.neckline
    assert sig.target < sig.entry
    assert sig.off_session
    assert qty_for_risk(sig.entry, sig.stop, 145) == 100


def test_weekend_signal_is_off_session_and_cash_session_is_not() -> None:
    weekend = detect_rollover_shorts(*_pair(_soxs_top()))
    # 5 分第 47 根的第一根 1 分 = 起點 + 235 分。15:05 UTC + 235 分 = 19:00 UTC = 美東 15:00。
    cash = detect_rollover_shorts(*_pair(_soxs_top(), start="2026-10-28 15:05"))
    assert len(weekend) == 1 and weekend[0].off_session
    assert len(cash) == 1 and not cash[0].off_session


def test_wick_through_neckline_does_not_enter() -> None:
    rows = _soxs_top()
    rows[47] = (52.8, 52.90, 51.90, 52.60, 5000)
    df5, df1 = _pair(rows)
    assert detect_rollover_shorts(df5, df1) == []


def test_close_above_ma25_does_not_enter() -> None:
    rows = _flat(40, 50.0)
    rows.extend(
        [
            (53.5, 55.00, 53.50, 54.50, 3000),
            (54.2, 54.20, 53.30, 53.60, 2000),
            (53.6, 53.50, 53.20, 53.40, 1500),
            (53.4, 53.80, 53.30, 53.60, 1200),
            (53.6, 54.10, 53.50, 53.90, 1200),
            (53.9, 53.90, 53.40, 53.60, 1100),
            (53.6, 53.70, 53.30, 53.50, 1100),
            (53.5, 53.40, 53.00, 53.05, 4000),
        ]
    )
    df5, df1 = _pair(rows)
    entry = 47 * 5
    for i in range(entry - 30, entry):
        _set_1m(df1, i, high=52.10, low=51.90, close=52.00, volume=100)
    _set_1m(df1, entry, high=53.10, low=52.90, close=53.05, volume=5000)
    assert detect_rollover_shorts(df5, df1) == []


def test_do_not_chase_once_price_has_left_the_neckline() -> None:
    rows = _soxs_top(break_close=50.50)
    rows[47] = (52.2, 52.40, 50.20, 50.50, 5000)
    rows.append((50.5, 51.20, 50.40, 51.00, 5000))
    df5, df1 = _pair(rows)
    assert detect_rollover_shorts(df5, df1) == []


def test_higher_bounce_under_the_extreme_lifts_the_stop() -> None:
    """較低高點之後又墊高一截、但仍低於極點，停損跟到新的較低高點，不跟到 54.85。"""

    rows = _soxs_top()
    rows[47] = (53.2, 53.80, 52.80, 53.60, 1000)
    rows.extend(
        [
            (53.6, 54.20, 53.00, 54.00, 1000),
            (54.0, 53.90, 53.20, 53.50, 1000),
            (53.5, 53.60, 53.00, 53.20, 1000),
            (53.2, 53.10, 51.90, 52.05, 4000),
        ]
    )
    df5, df1 = _pair(rows)
    signals = detect_rollover_shorts(df5, df1)
    assert len(signals) == 1, signals
    sig = signals[0]
    assert sig.kind == "break"
    assert sig.bar_idx == 51 * 5
    assert sig.entry == 52.05
    assert sig.neckline == 52.35
    assert sig.lower_high == 54.20
    assert sig.stop == 54.20
    assert sig.stop < sig.extreme


def test_new_high_invalidates_the_lower_high() -> None:
    rows = _soxs_top()
    rows[47] = (53.2, 55.30, 53.00, 55.00, 5000)
    rows.append((54.5, 54.60, 51.40, 51.70, 5000))
    df5, df1 = _pair(rows)
    assert detect_rollover_shorts(df5, df1) == []


def test_low_volume_break_allows_one_retest() -> None:
    df5, df1 = _pair(_soxs_top(break_volume=200))
    entry = 47 * 5
    for i in range(entry - 30, entry):
        _set_1m(df1, i, high=52.55, low=52.35, close=52.45, volume=200)
    _set_1m(df1, entry, high=52.20, low=51.95, close=52.05, volume=1)
    _set_1m(df1, entry + 1, high=52.55, low=52.10, close=52.25, volume=500)
    signals = detect_rollover_shorts(df5, df1)
    assert len(signals) == 1, signals
    sig = signals[0]
    assert sig.kind == "retest"
    assert sig.bar_idx == entry + 1
    assert sig.entry == 52.25
    assert sig.stop == 52.55
    assert sig.stop < sig.lower_high
    assert sig.extreme == 54.85
    assert sig.entry < sig.neckline


def test_rejects_other_timeframes() -> None:
    rows = _soxs_top()
    df5, df1 = _pair(rows)
    df15 = _frame(rows, "2026-10-31", "15min")
    try:
        detect_rollover_shorts(df15, df1)
    except ValueError as exc:
        assert "5 分" in str(exc)
    else:
        raise AssertionError("15 分不該拿來當結構")
    try:
        detect_rollover_shorts(df5, df5)
    except ValueError as exc:
        assert "1 分" in str(exc)
    else:
        raise AssertionError("5 分不該拿來當進場")


def test_inspect_names_the_break_and_the_chase() -> None:
    df5, df1 = _pair(_soxs_top())
    entry = 47 * 5
    caught = inspect_rollover(df5, df1.iloc[: entry + 1])
    assert caught.fresh is not None
    assert caught.fresh.kind == "break"
    assert caught.fresh.entry == 52.05
    assert "起跌" in caught.note
    assert caught.extreme == 54.85
    assert caught.neckline == 52.35

    chase5, chase1 = _pair(_soxs_top(break_close=50.50) + [(50.5, 51.20, 50.40, 51.00, 5000)])
    missed = inspect_rollover(chase5, chase1)
    assert missed.fresh is None
    assert missed.signals == ()
    assert "不追" in missed.note

    import soxs_rollover

    text = soxs_rollover.format_report("SOXSUSDT", caught, 100)
    assert "抓到起跌" in text
    assert "52.05" in text
    assert "53.5" in text


def test_closed_klines_drop_the_forming_bar() -> None:
    import soxs_rollover

    raw = [
        [1_700_000_000_000, "10", "11", "9", "10.5", "100"],
        [1_700_000_060_000, "10.5", "12", "10", "11", "120"],
    ]
    frame = soxs_rollover.frame_from_klines(raw, "1m", now_ms=1_700_000_060_000 + 10_000)
    assert len(frame) == 1
    assert float(frame.iloc[-1].close) == 10.5
    kept = soxs_rollover.frame_from_klines(raw, "1m", now_ms=1_700_000_120_000)
    assert len(kept) == 2


def test_simulate_target_stop_and_same_bar_stop_wins() -> None:
    def _with_extra(extra: tuple) -> tuple[pd.DataFrame, list]:
        df5, df1 = _pair(_soxs_top() + [extra])
        return df1, detect_rollover_shorts(df5, df1)

    down_1m, down_sigs = _with_extra((52.0, 52.40, 49.40, 49.70, 2000))
    up_1m, up_sigs = _with_extra((52.0, 53.80, 51.80, 53.20, 2000))
    target_trades = simulate_shorts(down_1m, down_sigs)
    stop_trades = simulate_shorts(up_1m, up_sigs)
    assert len(target_trades) == 1 and target_trades[0].exit_reason == "target"
    assert target_trades[0].pnl_per_unit > 0
    assert len(stop_trades) == 1 and stop_trades[0].exit_reason == "stop"
    assert stop_trades[0].pnl_per_unit < 0

    index = pd.date_range("2026-10-28 14:00", periods=4, freq="1min", tz="UTC")
    both = pd.DataFrame(
        {
            "open": [50, 50, 50, 50],
            "high": [50.2, 51.2, 50.2, 50.2],
            "low": [49.8, 40.0, 49.8, 49.8],
            "close": [50, 45, 50, 50],
            "volume": [1, 1, 1, 1],
        },
        index=index,
    )
    signal = RolloverSignal(
        timestamp=index[0],
        bar_idx=0,
        kind="break",
        entry=50.0,
        stop=51.0,
        target=47.0,
        extreme=54.0,
        extreme_idx=0,
        neckline=49.0,
        lower_high=51.0,
        lower_high_idx=0,
        off_session=False,
    )
    trades = simulate_shorts(both, [signal], RolloverParams(max_hold_bars=2))
    assert trades[0].exit_reason == "stop"
    assert trades[0].exit_idx == 1


def main() -> int:
    test_us_cash_session()
    test_qty_uses_stop_distance()
    test_uptrend_sell_marks_do_not_short()
    test_break_close_stops_at_lower_high_not_extreme()
    test_weekend_signal_is_off_session_and_cash_session_is_not()
    test_wick_through_neckline_does_not_enter()
    test_close_above_ma25_does_not_enter()
    test_do_not_chase_once_price_has_left_the_neckline()
    test_higher_bounce_under_the_extreme_lifts_the_stop()
    test_new_high_invalidates_the_lower_high()
    test_low_volume_break_allows_one_retest()
    test_rejects_other_timeframes()
    test_inspect_names_the_break_and_the_chase()
    test_closed_klines_drop_the_forming_bar()
    test_simulate_target_stop_and_same_bar_stop_wins()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

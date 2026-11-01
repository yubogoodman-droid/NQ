#!/usr/bin/env python3
"""合成 15 分 K 測試起跌放空。不打網路。

走勢照 2026-10-31 SOXS 那一段：先墊高，極點 54.85，再以較低高點跌破頸線。
極點本身不是進場。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rollover.strategy import (  # noqa: E402
    RolloverParams,
    RolloverSignal,
    detect_rollover_shorts,
    is_us_cash_session,
    qty_for_risk,
    simulate_shorts,
)


def _frame(rows: list[tuple], start: str = "2026-10-31 00:00") -> pd.DataFrame:
    index = pd.date_range(start, periods=len(rows), freq="15min", tz="UTC")
    return pd.DataFrame(rows, index=index, columns=["open", "high", "low", "close", "volume"])


def _flat(n: int, price: float, volume: float = 1000.0) -> list[tuple]:
    return [(price, price + 0.10, price - 0.10, price, volume) for _ in range(n)]


def _soxs_top(break_volume: float = 4000.0, break_close: float = 52.05) -> list[tuple]:
    """40 根橫在 52，再衝 54.85，回檔頸線 52.35，較低高點 53.50，第 47 根嘗試跌破。"""

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
    assert detect_rollover_shorts(_frame(rows)) == []


def test_break_close_stops_at_lower_high_not_extreme() -> None:
    df = _frame(_soxs_top())
    signals = detect_rollover_shorts(df)
    assert len(signals) == 1, signals
    sig = signals[0]
    assert sig.kind == "break"
    assert sig.bar_idx == 47
    assert sig.entry == 52.05
    assert sig.neckline == 52.35
    assert sig.extreme == 54.85
    assert sig.lower_high == 53.50
    assert sig.stop == 53.50
    assert sig.stop < sig.extreme
    assert sig.entry < sig.neckline
    assert sig.target < sig.entry
    assert sig.off_session
    assert qty_for_risk(sig.entry, sig.stop, 145) == 100


def test_weekend_signal_is_off_session_and_cash_session_is_not() -> None:
    weekend = detect_rollover_shorts(_frame(_soxs_top()))
    cash = detect_rollover_shorts(_frame(_soxs_top(), start="2026-10-28 07:15"))
    assert len(weekend) == 1 and weekend[0].off_session
    assert len(cash) == 1 and not cash[0].off_session


def test_wick_through_neckline_does_not_enter() -> None:
    rows = _soxs_top()
    rows[47] = (52.8, 52.90, 51.90, 52.60, 5000)
    assert detect_rollover_shorts(_frame(rows)) == []


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
    assert detect_rollover_shorts(_frame(rows)) == []


def test_do_not_chase_once_price_has_left_the_neckline() -> None:
    rows = _soxs_top(break_close=50.50)
    rows[47] = (52.2, 52.40, 50.20, 50.50, 5000)
    rows.append((50.5, 51.20, 50.40, 51.00, 5000))
    assert detect_rollover_shorts(_frame(rows)) == []


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
    signals = detect_rollover_shorts(_frame(rows))
    assert len(signals) == 1, signals
    sig = signals[0]
    assert sig.kind == "break"
    assert sig.bar_idx == 51
    assert sig.entry == 52.05
    assert sig.neckline == 52.35
    assert sig.lower_high == 54.20
    assert sig.stop == 54.20
    assert sig.stop < sig.extreme


def test_new_high_invalidates_the_lower_high() -> None:
    rows = _soxs_top()
    rows[47] = (53.2, 55.30, 53.00, 55.00, 5000)
    rows.append((54.5, 54.60, 51.40, 51.70, 5000))
    assert detect_rollover_shorts(_frame(rows)) == []


def test_low_volume_break_allows_one_retest() -> None:
    rows = _soxs_top(break_volume=200)
    rows.append((52.20, 52.55, 52.00, 52.15, 800))
    rows.append((52.15, 52.40, 51.70, 51.80, 4000))
    signals = detect_rollover_shorts(_frame(rows))
    assert len(signals) == 1, signals
    sig = signals[0]
    assert sig.kind == "retest"
    assert sig.bar_idx == 48
    assert sig.entry == 52.15
    assert sig.stop == 52.70
    assert sig.stop < sig.lower_high
    assert sig.extreme == 54.85
    assert sig.entry < sig.neckline


def test_rejects_bars_that_are_not_15m() -> None:
    rows = _soxs_top()
    index = pd.date_range("2026-10-31", periods=len(rows), freq="5min", tz="UTC")
    df = pd.DataFrame(rows, index=index, columns=["open", "high", "low", "close", "volume"])
    try:
        detect_rollover_shorts(df)
    except ValueError as exc:
        assert "15 分" in str(exc)
    else:
        raise AssertionError("5 分 K 不該被當成起跌確認")


def test_simulate_target_stop_and_same_bar_stop_wins() -> None:
    base = _soxs_top()
    down = base + [(52.0, 52.40, 49.40, 49.70, 2000)]
    up = base + [(52.0, 53.80, 51.80, 53.20, 2000)]
    target_trades = simulate_shorts(_frame(down), detect_rollover_shorts(_frame(down)))
    stop_trades = simulate_shorts(_frame(up), detect_rollover_shorts(_frame(up)))
    assert len(target_trades) == 1 and target_trades[0].exit_reason == "target"
    assert target_trades[0].pnl_per_unit > 0
    assert len(stop_trades) == 1 and stop_trades[0].exit_reason == "stop"
    assert stop_trades[0].pnl_per_unit < 0

    index = pd.date_range("2026-10-28 14:00", periods=4, freq="15min", tz="UTC")
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
    test_rejects_bars_that_are_not_15m()
    test_simulate_target_stop_and_same_bar_stop_wins()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

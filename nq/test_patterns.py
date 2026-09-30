"""三小時破底後 MA5/MA20 多排站上 MA60 測試。"""

from __future__ import annotations

import pandas as pd

from nq.patterns import detect_reclaims
from nq.strategy import NQWBottomStrategy


def _df(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    data = []
    for o, h, l, c in rows:
        data.append(
            {
                "open": o,
                "high": max(o, h, c),
                "low": min(o, l, c),
                "close": c,
                "volume": 200,
            }
        )
    idx = pd.date_range("2026-08-01 09:30", periods=len(data), freq="5min")
    return pd.DataFrame(data, index=idx)


def _flat(n: int, price: float, drift: float = 0.0) -> list[tuple[float, float, float, float]]:
    rows = []
    px = price
    for _ in range(n):
        px = px + drift
        rows.append((px, px + 4, px - 4, px))
    return rows


def _break_then_stack_rows() -> list[tuple[float, float, float, float]]:
    """破底後快速拉升，讓 MA5>MA20>MA60 且收盤在 MA60 上。"""
    rows = _flat(65, 10040.0)
    rows.append((10020, 10024, 9990, 10000))
    rows.extend(_flat(8, 10080.0))
    return rows


def test_detects_5_20_stack_above_ma60() -> None:
    df = _df(_break_then_stack_rows())
    patterns = detect_reclaims(df)
    assert len(patterns) >= 1
    p = patterns[0]
    assert p.break_low <= 9990
    assert p.ma5 > p.ma20 > p.ma60
    assert df["close"].iloc[p.entry_idx] > p.ma60
    assert p.entry_idx - p.break_idx < 12


def test_no_stack_within_one_hour_is_ignored() -> None:
    rows = _flat(65, 10040.0)
    rows.append((10020, 10024, 9990, 9995))
    rows.extend(_flat(14, 9992.0))
    assert detect_reclaims(_df(rows)) == []


def test_no_break_of_3h_low_is_ignored() -> None:
    assert detect_reclaims(_df(_flat(80, 10040.0))) == []


def test_strategy_enters_on_stack_close() -> None:
    df = _df(_break_then_stack_rows())
    signals = NQWBottomStrategy().generate_signals(df)
    assert len(signals) == 1
    sig = signals[0]
    assert sig.bar_idx == sig.pattern.entry_idx
    assert sig.stop_loss < sig.entry < sig.target

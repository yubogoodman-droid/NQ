"""三小時破底翻 MA30 測試。"""

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


def _break_then_reclaim_rows() -> list[tuple[float, float, float, float]]:
    """前 40 根在 10040，接著一根跌破三小時低，兩根內收盤站上 MA30。"""
    rows = _flat(40, 10040.0)
    rows.append((10020, 10024, 9990, 10010))  # 破 10036 附近的三小時低
    rows.append((10010, 10050, 10008, 10045))  # 收盤站上 MA30
    rows.extend(_flat(8, 10048.0, drift=1.0))
    return rows


def test_detects_break_and_ma30_reclaim() -> None:
    df = _df(_break_then_reclaim_rows())
    patterns = detect_reclaims(df)
    assert len(patterns) >= 1
    p = patterns[0]
    assert p.break_low <= 9990
    assert p.entry_idx > p.break_idx
    assert df["close"].iloc[p.entry_idx] > p.ma30


def test_no_reclaim_within_30_minutes_is_ignored() -> None:
    rows = _flat(40, 10040.0)
    rows.append((10020, 10024, 9990, 9995))
    rows.extend(_flat(8, 9992.0))  # 一直趴在均線下
    assert detect_reclaims(_df(rows)) == []


def test_no_break_of_3h_low_is_ignored() -> None:
    rows = _flat(50, 10040.0)
    assert detect_reclaims(_df(rows)) == []


def test_strategy_enters_on_reclaim_close() -> None:
    df = _df(_break_then_reclaim_rows())
    signals = NQWBottomStrategy().generate_signals(df)
    assert len(signals) == 1
    sig = signals[0]
    assert sig.bar_idx == sig.pattern.entry_idx
    assert sig.entry == round(float(df["close"].iloc[sig.bar_idx]) / 0.25) * 0.25
    assert sig.stop_loss < sig.entry
    assert sig.target > sig.entry

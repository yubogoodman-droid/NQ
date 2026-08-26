"""破底 W 底偵測測試。"""

from __future__ import annotations

import pandas as pd

from nq.patterns import detect_w_bottoms
from nq.strategy import NQWBottomStrategy


def _df(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    data = []
    for o, h, l, c in rows:
        hi = max(o, h, c)
        lo = min(o, l, c)
        data.append({"open": o, "high": hi, "low": lo, "close": c, "volume": 200})
    idx = pd.date_range("2026-08-01 09:30", periods=len(data), freq="5min")
    return pd.DataFrame(data, index=idx)


def _flat(n: int, price: float, drift: float = 0.0) -> list[tuple[float, float, float, float]]:
    rows = []
    px = price
    for _ in range(n):
        px = px + drift
        rows.append((px, px + 4, px - 4, px))
    return rows


def _break_low_w_rows() -> list[tuple[float, float, float, float]]:
    """L1 → 頸線反彈 → L2 跌破 L1 → 收復後直接突破，沒有第三隻腳。"""
    rows: list[tuple[float, float, float, float]] = []
    rows.extend(_flat(8, 10040.0))
    rows.append((10020, 10024, 10012, 10016))
    rows.append((10016, 10018, 10000, 10008))  # L1 = 10000
    rows.append((10008, 10022, 10006, 10020))
    rows.append((10020, 10032, 10016, 10030))
    rows.append((10030, 10042, 10026, 10040))
    rows.append((10040, 10052, 10036, 10048))  # 頸線 10052
    rows.append((10048, 10050, 10030, 10032))
    rows.append((10032, 10034, 9970, 9978))  # L2 破底
    rows.append((9978, 10012, 9968, 10008))  # 收復 L1，L2 swing low = 9968
    rows.append((10008, 10024, 10004, 10022))
    rows.append((10022, 10038, 10018, 10036))
    rows.append((10036, 10048, 10032, 10046))
    rows.append((10046, 10060, 10044, 10056))  # 突破頸線
    rows.extend(_flat(6, 10058.0, drift=1.0))
    return rows


def test_detects_two_leg_w_with_right_foot_breakdown() -> None:
    df = _df(_break_low_w_rows())
    patterns = detect_w_bottoms(df)
    assert len(patterns) >= 1
    p = patterns[0]
    assert p.first_low == 10000
    assert p.second_low < p.first_low
    assert p.second_low <= 9970
    assert p.spring_idx == p.second_low_idx
    assert p.breakout_idx is not None
    assert df["close"].iloc[p.breakout_idx] > p.neckline
    assert p.stop_loss == p.second_low


def test_equal_double_bottom_without_breakdown_is_ignored() -> None:
    rows = _flat(8, 10040.0)
    rows.append((10020, 10024, 10012, 10016))
    rows.append((10016, 10018, 10000, 10008))  # L1
    rows.append((10008, 10022, 10006, 10020))
    rows.append((10020, 10032, 10016, 10030))
    rows.append((10030, 10042, 10026, 10040))
    rows.append((10040, 10052, 10036, 10048))
    rows.append((10048, 10050, 10030, 10032))
    rows.append((10032, 10036, 10008, 10012))
    rows.append((10012, 10020, 10006, 10016))
    rows.append((10016, 10018, 10001, 10008))  # L2 ≈ L1，沒跌破
    rows.append((10008, 10022, 10006, 10020))
    rows.append((10020, 10036, 10016, 10034))
    rows.append((10034, 10048, 10030, 10046))
    rows.append((10046, 10060, 10044, 10056))
    rows.extend(_flat(6, 10058.0))
    assert detect_w_bottoms(_df(rows)) == []


def test_breakdown_without_reclaim_is_ignored() -> None:
    rows = _flat(8, 10040.0)
    rows.append((10020, 10024, 10012, 10016))
    rows.append((10016, 10018, 10000, 10008))
    rows.append((10008, 10022, 10006, 10020))
    rows.append((10020, 10032, 10016, 10030))
    rows.append((10030, 10042, 10026, 10040))
    rows.append((10040, 10052, 10036, 10048))
    rows.append((10032, 10034, 9970, 9978))
    rows.extend(_flat(16, 9960.0, drift=-2.0))
    assert detect_w_bottoms(_df(rows)) == []


def test_strategy_emits_long_on_neckline_break() -> None:
    df = _df(_break_low_w_rows())
    signals = NQWBottomStrategy().generate_signals(df)
    assert len(signals) == 1
    sig = signals[0]
    assert sig.entry > sig.stop_loss
    assert sig.target > sig.entry
    assert sig.pattern.second_low < sig.pattern.first_low

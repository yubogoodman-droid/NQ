"""破底 W 底偵測測試。"""

from __future__ import annotations

import pandas as pd

from nq.patterns import detect_w_bottoms
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


def _l1_l2_l3_rows() -> list[tuple[float, float, float, float]]:
    """L1 左 → 頸線 → L2 破底 → 收復 → L3 右 → 突破。"""
    rows: list[tuple[float, float, float, float]] = []
    rows.extend(_flat(8, 10040.0))
    rows.append((10020, 10024, 10012, 10016))
    rows.append((10016, 10018, 10000, 10008))  # L1
    rows.append((10008, 10022, 10006, 10020))
    rows.append((10020, 10032, 10016, 10030))
    rows.append((10030, 10042, 10026, 10040))
    rows.append((10040, 10052, 10036, 10048))  # 頸線
    rows.append((10048, 10050, 10030, 10032))
    rows.append((10032, 10034, 9970, 9978))  # L2 破底開始
    rows.append((9978, 10012, 9968, 10008))  # L2 最低 + 收復
    rows.append((10008, 10020, 10004, 10016))
    rows.append((10016, 10022, 10006, 10010))
    rows.append((10010, 10014, 10001, 10006))  # L3 ≈ L1
    rows.append((10006, 10018, 10004, 10016))
    rows.append((10016, 10028, 10012, 10026))
    rows.append((10026, 10038, 10022, 10036))
    rows.append((10036, 10048, 10032, 10046))
    rows.append((10046, 10060, 10044, 10056))
    rows.extend(_flat(6, 10058.0, drift=1.0))
    return rows


def test_detects_l1_l2_breakdown_l3() -> None:
    df = _df(_l1_l2_l3_rows())
    patterns = detect_w_bottoms(df)
    assert len(patterns) >= 1
    p = patterns[0]
    assert p.l1 == 10000
    assert p.l2 < p.l1
    assert p.l2 <= 9970
    assert abs(p.l3 - p.l1) / p.l1 <= 0.001
    assert p.l3 > p.l2
    assert p.l1_idx < p.l2_idx < p.l3_idx
    assert p.breakout_idx is not None
    assert df["close"].iloc[p.breakout_idx] > p.neckline
    assert p.stop_loss == p.l3


def test_l2_is_the_lowest_bar_between_shoulders() -> None:
    """中間若有更深的殺，L2 必須抓那根，不能抓後來略破 L1 的回測。"""
    rows = _flat(8, 10040.0)
    rows.append((10020, 10024, 10012, 10016))
    rows.append((10016, 10018, 10000, 10008))  # L1 = 10000
    rows.append((10008, 10022, 10006, 10020))
    rows.append((10020, 10032, 10016, 10030))
    rows.append((10030, 10042, 10026, 10040))
    rows.append((10040, 10052, 10036, 10048))  # 先彈出頸線
    rows.append((10048, 10050, 10030, 10032))
    rows.append((10030, 10018, 9950, 9955))  # 真正破底
    rows.append((9955, 10020, 9948, 10010))  # 最低 9948 並收復
    rows.append((10010, 10040, 10004, 10036))
    rows.append((10036, 10048, 10020, 10028))
    rows.append((10028, 10030, 9992, 10008))  # 淺回測，不可當 L2
    rows.append((10008, 10018, 10002, 10012))
    rows.append((10012, 10016, 10001, 10008))  # L3 ≈ L1
    rows.append((10008, 10022, 10004, 10018))
    rows.append((10018, 10034, 10014, 10032))
    rows.append((10032, 10048, 10028, 10046))
    rows.append((10046, 10060, 10044, 10056))
    rows.extend(_flat(6, 10058.0, drift=1.0))
    patterns = detect_w_bottoms(_df(rows))
    assert patterns
    p = patterns[0]
    assert p.l2 <= 9950
    assert p.l2 < 9992


def test_two_leg_without_right_shoulder_is_ignored() -> None:
    rows = _flat(8, 10040.0)
    rows.append((10020, 10024, 10012, 10016))
    rows.append((10016, 10018, 10000, 10008))
    rows.append((10008, 10022, 10006, 10020))
    rows.append((10020, 10032, 10016, 10030))
    rows.append((10030, 10042, 10026, 10040))
    rows.append((10040, 10052, 10036, 10048))
    rows.append((10048, 10050, 10030, 10032))
    rows.append((10032, 10034, 9970, 9978))
    rows.append((9978, 10012, 9968, 10008))
    rows.append((10008, 10024, 10004, 10022))
    rows.append((10022, 10038, 10018, 10036))
    rows.append((10036, 10048, 10032, 10046))
    rows.append((10046, 10060, 10044, 10056))
    rows.extend(_flat(6, 10058.0, drift=1.0))
    assert detect_w_bottoms(_df(rows)) == []


def test_equal_double_bottom_without_l2_breakdown_is_ignored() -> None:
    rows = _flat(8, 10040.0)
    rows.append((10020, 10024, 10012, 10016))
    rows.append((10016, 10018, 10000, 10008))
    rows.append((10008, 10022, 10006, 10020))
    rows.append((10020, 10032, 10016, 10030))
    rows.append((10030, 10042, 10026, 10040))
    rows.append((10040, 10052, 10036, 10048))
    rows.append((10048, 10050, 10030, 10032))
    rows.append((10032, 10036, 10008, 10012))
    rows.append((10012, 10020, 10006, 10016))
    rows.append((10016, 10018, 10001, 10008))
    rows.append((10008, 10022, 10006, 10020))
    rows.append((10020, 10036, 10016, 10034))
    rows.append((10034, 10048, 10030, 10046))
    rows.append((10046, 10060, 10044, 10056))
    rows.extend(_flat(6, 10058.0))
    assert detect_w_bottoms(_df(rows)) == []


def test_strategy_emits_long_on_neckline_break() -> None:
    df = _df(_l1_l2_l3_rows())
    signals = NQWBottomStrategy().generate_signals(df)
    assert len(signals) == 1
    sig = signals[0]
    assert sig.pattern.l1_idx < sig.pattern.l2_idx < sig.pattern.l3_idx
    assert sig.pattern.l2 < sig.pattern.l1
    assert sig.entry > sig.stop_loss

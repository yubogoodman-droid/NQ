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
    for i in range(n):
        px = px + drift
        rows.append((px, px + 4, px - 4, px))
    return rows


def _spring_w_rows() -> list[tuple[float, float, float, float]]:
    """L1 → 反彈 → 破底 → 收復 → L2 → 突破頸線。"""
    rows: list[tuple[float, float, float, float]] = []
    rows.extend(_flat(8, 10040.0))
    # L1 at index 10 (after 8 flats + 2 decline)
    rows.append((10020, 10024, 10012, 10016))
    rows.append((10016, 10018, 10000, 10008))  # L1 low=10000
    # right-side confirmation + bounce to neckline ~10050
    rows.append((10008, 10022, 10006, 10020))
    rows.append((10020, 10032, 10016, 10030))
    rows.append((10030, 10042, 10026, 10040))
    rows.append((10040, 10052, 10036, 10048))  # bounce high 10052
    rows.append((10048, 10050, 10030, 10032))
    # 破底：低於 L1 約 30 點
    rows.append((10032, 10034, 9970, 9978))
    rows.append((9978, 10012, 9968, 10008))  # 收復 close > L1, spring low 9968
    # 小彈後 L2 ≈ L1
    rows.append((10008, 10020, 10004, 10016))
    rows.append((10016, 10022, 10006, 10010))
    rows.append((10010, 10014, 10001, 10006))  # L2 low=10001
    rows.append((10006, 10018, 10004, 10016))
    rows.append((10016, 10028, 10012, 10026))
    rows.append((10026, 10038, 10022, 10036))
    # 突破頸線 10052
    rows.append((10036, 10048, 10032, 10046))
    rows.append((10046, 10060, 10044, 10056))
    rows.extend(_flat(6, 10058.0, drift=1.0))
    return rows


def test_detects_spring_w_with_middle_breakdown() -> None:
    df = _df(_spring_w_rows())
    patterns = detect_w_bottoms(df)
    assert len(patterns) >= 1
    p = patterns[0]
    assert p.first_low == 10000
    assert p.spring_low <= 9970
    assert abs(p.second_low - p.first_low) / p.first_low <= 0.001
    assert p.second_low > p.spring_low
    assert p.breakout_idx is not None
    assert df["close"].iloc[p.breakout_idx] > p.neckline
    assert p.target > p.neckline
    assert p.stop_loss == p.second_low


def test_vanilla_double_bottom_without_spring_is_ignored() -> None:
    rows = _flat(8, 10040.0)
    rows.append((10020, 10024, 10012, 10016))
    rows.append((10016, 10018, 10000, 10008))  # L1
    rows.append((10008, 10022, 10006, 10020))
    rows.append((10020, 10032, 10016, 10030))
    rows.append((10030, 10042, 10026, 10040))
    rows.append((10040, 10052, 10036, 10048))
    rows.append((10048, 10050, 10030, 10032))
    rows.append((10032, 10036, 10008, 10012))  # 回測，沒破 L1
    rows.append((10012, 10020, 10006, 10016))
    rows.append((10016, 10018, 10001, 10008))  # L2 ≈ L1，沒有破底
    rows.append((10008, 10022, 10006, 10020))
    rows.append((10020, 10036, 10016, 10034))
    rows.append((10034, 10048, 10030, 10046))
    rows.append((10046, 10060, 10044, 10056))
    rows.extend(_flat(6, 10058.0))
    df = _df(rows)
    assert detect_w_bottoms(df) == []


def test_spring_without_reclaim_is_ignored() -> None:
    rows = _flat(8, 10040.0)
    rows.append((10020, 10024, 10012, 10016))
    rows.append((10016, 10018, 10000, 10008))  # L1
    rows.append((10008, 10022, 10006, 10020))
    rows.append((10020, 10032, 10016, 10030))
    rows.append((10030, 10042, 10026, 10040))
    rows.append((10040, 10052, 10036, 10048))
    rows.append((10032, 10034, 9970, 9978))  # 破底
    rows.extend(_flat(16, 9960.0, drift=-2.0))  # 沒有收復
    df = _df(rows)
    assert detect_w_bottoms(df) == []


def test_strategy_emits_long_on_neckline_break() -> None:
    df = _df(_spring_w_rows())
    signals = NQWBottomStrategy().generate_signals(df)
    assert len(signals) == 1
    sig = signals[0]
    assert sig.entry > sig.stop_loss
    assert sig.target > sig.entry
    assert sig.pattern.spring_low < sig.pattern.first_low

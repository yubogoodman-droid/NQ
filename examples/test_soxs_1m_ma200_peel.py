#!/usr/bin/env python3
"""SOXS 一分 K 撥開進場的離線測試。不打幣安。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from soxs_1m_ma200_peel import (  # noqa: E402
    entry_indices,
    is_peel,
    simulate,
)


def _flat(n: int = 80, px: float = 100.0):
    c = np.full(n, px)
    h = np.full(n, px + 0.05)
    l = np.full(n, px - 0.05)
    m7 = np.full(n, px)
    m14 = np.full(n, px)
    m25 = np.full(n, px)
    m99 = np.full(n, px)
    m120 = np.full(n, px)
    m200 = np.full(n, px)
    t = np.arange(n, dtype=np.int64) * 60_000
    return c, h, l, m7, m14, m25, m99, m120, m200, t


def _arm_peel(i: int = 50):
    """前段纏在 MA200 上，第 i 根短均線剛撥開、收盤貼著 MA200。"""
    c, h, l, m7, m14, m25, m99, m120, m200, t = _flat(i + 5)
    # 前 8 根裡要有一根還纏著。i-6 維持平的，價差 0。
    c[i] = 100.20
    h[i] = 100.28
    l[i] = 99.90
    m7[i] = 100.15
    m14[i] = 100.00
    m25[i] = 99.80
    m7[i - 5] = 99.70
    m14[i - 5] = 99.80
    m25[i - 5] = 99.70
    # 斜率：m7 約 +0.45%，m14 約 +0.20%，m25 約 +0.10%
    return c, h, l, m7, m14, m25, m99, m120, m200, t


def test_peel_bar_passes_and_fan_fails() -> None:
    c, h, l, m7, m14, m25, m99, m120, m200, _t = _arm_peel(50)
    assert is_peel(50, c, l, m7, m14, m25, m99, m120, m200)
    assert not is_peel(49, c, l, m7, m14, m25, m99, m120, m200)

    # 扇形已經張開、價格離開 MA200：不是起漲
    c2 = c.copy()
    m7b, m14b, m25b = m7.copy(), m14.copy(), m25.copy()
    c2[50] = 103.0
    m7b[50], m14b[50], m25b[50] = 102.4, 101.5, 100.8
    assert not is_peel(50, c2, l, m7b, m14b, m25b, m99, m120, m200)

    # MA200 末端已經上弧
    m200b = m200.copy()
    m200b[50] = 100.40
    m200b[40] = 100.0
    assert not is_peel(50, c, l, m7, m14, m25, m99, m120, m200b)


def test_one_shot_until_back_under() -> None:
    c, h, l, m7, m14, m25, m99, m120, m200, t = _arm_peel(60)
    # 讓第 40 根之前先有 15 根在 MA200 下，武裝進場
    c[:20] = 99.0
    assert is_peel(60, c, l, m7, m14, m25, m99, m120, m200)
    # 緊接著再做一根一樣的，不該再進
    for j in (61, 62):
        c[j] = c[60]
        l[j] = l[60]
        h[j] = h[60]
        m7[j], m14[j], m25[j] = m7[60], m14[60], m25[60]
        m7[j - 5], m14[j - 5], m25[j - 5] = m7[55], m14[55], m25[55]
    idxs = entry_indices(c, l, m7, m14, m25, m99, m120, m200, t, t_min=0)
    assert idxs == [60], idxs


def test_rearm_after_fifteen_closes_under() -> None:
    c, h, l, m7, m14, m25, m99, m120, m200, t = _flat(120)
    c[:20] = 99.0

    def plant(i: int) -> None:
        c[i] = 100.20
        h[i] = 100.28
        l[i] = 99.90
        m7[i], m14[i], m25[i] = 100.15, 100.00, 99.80
        m7[i - 5], m14[i - 5], m25[i - 5] = 99.70, 99.80, 99.70

    plant(40)
    # 41..55 收在 MA200 下，滿 15 根後才准下一筆
    c[41:56] = 99.5
    plant(70)
    idxs = entry_indices(c, l, m7, m14, m25, m99, m120, m200, t, t_min=0)
    assert idxs == [40, 70], idxs


def test_exits_stop_before_target_and_scale() -> None:
    n = 30
    c = np.full(n, 100.0)
    h = np.full(n, 100.1)
    l = np.full(n, 99.8)
    m200 = np.full(n, 100.0)
    # 前 40 根不夠，simulate 用 i-40。把進場放在 10，前面低點做停損。
    # BASE_BARS 是 40，所以 i 必須 >= 40。
    c = np.full(50, 100.0)
    h = np.full(50, 100.05)
    l = np.full(50, 99.9)
    m200 = np.full(50, 100.0)
    l[:40] = 99.0  # 停損 99.0
    i = 40
    c[i] = 100.2
    # 下一根同時打到停損和 +1：算停損
    l[41] = 98.5
    h[41] = 101.5
    c[41] = 100.0
    tr = simulate(i, h, l, c, m200)
    assert tr["scalp_reason"] == "停損"
    assert tr["scale_reason"] == "停損"
    assert abs(tr["scalp_pts"] - (99.0 - 100.2)) < 1e-9

    # 先 +1，之後收盤跌回 MA200：整筆 +1；一半鎖 1 點，另一半在跌回價出場
    c2 = np.full(50, 100.2)
    h2 = np.full(50, 100.3)
    l2 = np.full(50, 99.5)
    l2[:40] = 99.0
    m2 = np.full(50, 100.0)
    c2[40] = 100.2
    h2[41] = 101.4  # +1 觸價
    l2[41] = 100.0
    c2[41] = 101.1
    c2[42] = 100.6
    h2[42] = 100.8
    l2[42] = 100.4
    c2[43] = 99.7  # 跌回 MA200
    h2[43] = 100.2
    l2[43] = 99.6
    tr2 = simulate(40, h2, l2, c2, m2)
    assert tr2["scalp_reason"] == "+1"
    assert abs(tr2["scalp_pts"] - 1.0) < 1e-9
    assert tr2["scale_reason"] == "一半後跌回"
    # 0.5*1 + 0.5*(99.7-100.2) = 0.5 - 0.25 = 0.25
    assert abs(tr2["scale_pts"] - 0.25) < 1e-9
    assert tr2["hold"] == 1
    assert tr2["scale_hold"] == 3


if __name__ == "__main__":
    test_peel_bar_passes_and_fan_fails()
    test_one_shot_until_back_under()
    test_rearm_after_fifteen_closes_under()
    test_exits_stop_before_target_and_scale()
    print("ALL TESTS PASSED")

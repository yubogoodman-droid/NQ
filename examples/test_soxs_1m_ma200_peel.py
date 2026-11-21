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


def _base_arrays(n: int = 120):
    c = np.full(n, 99.0)
    h = np.full(n, 99.1)
    l = np.full(n, 98.9)
    m7 = np.full(n, 99.0)
    m14 = np.full(n, 99.0)
    m25 = np.full(n, 99.0)
    m30 = np.full(n, 99.0)
    m99 = np.full(n, 99.2)
    m120 = np.full(n, 99.5)
    m200 = np.full(n, 100.0)
    t = np.arange(n, dtype=np.int64) * 60_000
    return c, h, l, m7, m14, m25, m30, m99, m120, m200, t


def _plant_peel(c, h, l, m7, m14, m25, m30, m99, m120, m200, i: int) -> None:
    """在第 i 根擺出 MA7 剛上穿平 MA200、連續站穩 4 根、長均線 200>120>99 的形狀。"""
    c[i - 3 : i + 1] = [100.10, 100.15, 100.25, 100.40]
    h[i - 3 : i + 1] = c[i - 3 : i + 1] + 0.08
    l[i - 3 : i + 1] = c[i - 3 : i + 1] - 0.08
    m7[i - 5] = 99.55
    m7[i - 1] = 99.95
    m7[i] = 100.15
    m14[i - 5] = 99.45
    m14[i] = 99.85
    m25[i - 5] = 99.40
    m25[i] = 99.60
    m30[i - 5] = 99.38
    m30[i] = 99.45
    m99[i] = 99.20
    m120[i] = 99.50
    m200[i] = 100.00


def test_peel_bar_passes_and_invalid_shapes_fail() -> None:
    c, h, l, m7, m14, m25, m30, m99, m120, m200, _t = _base_arrays(80)
    _plant_peel(c, h, l, m7, m14, m25, m30, m99, m120, m200, 55)
    assert is_peel(55, c, m7, m14, m25, m30, m99, m120, m200)
    assert not is_peel(54, c, m7, m14, m25, m30, m99, m120, m200)

    # 價格已經離開 MA200 超過 1%：不追
    c2 = c.copy()
    c2[55] = 101.5
    assert not is_peel(55, c2, m7, m14, m25, m30, m99, m120, m200)

    # 長均線不是 200 > 120 > 99（打結/翻多震盪）：過濾掉
    m120b = m120.copy()
    m120b[55] = 100.2
    assert not is_peel(55, c, m7, m14, m25, m30, m99, m120b, m200)

    # MA25 還躺平（10 根斜率不到 0.25%）
    m25b = m25.copy()
    m25b[45] = 99.45
    assert not is_peel(55, c, m7, m14, m25b, m30, m99, m120, m200)

    # MA200 末端已經上弧
    m200b = m200.copy()
    m200b[55] = 100.30
    m200b[45] = 100.0
    assert not is_peel(55, c, m7, m14, m25, m30, m99, m120, m200b)


def test_one_shot_and_rearm_after_25_under() -> None:
    c, h, l, m7, m14, m25, m30, m99, m120, m200, t = _base_arrays(160)
    _plant_peel(c, h, l, m7, m14, m25, m30, m99, m120, m200, 55)
    # 56 根就算也符合，也不該連進
    _plant_peel(c, h, l, m7, m14, m25, m30, m99, m120, m200, 56)
    c[52:57] = [100.10, 100.15, 100.25, 100.40, 100.45]
    m7[54] = 99.95
    m7[55] = 100.15
    m7[56] = 100.20

    # 57..115 都在 MA200 下（遠超 25 根）後，120 才准下一筆
    c[57:117] = 99.0
    _plant_peel(c, h, l, m7, m14, m25, m30, m99, m120, m200, 120)
    idxs = entry_indices(c, m7, m14, m25, m30, m99, m120, m200, t, t_min=0)
    assert idxs == [55, 120], idxs


def test_exits_buffer_stop_and_scale() -> None:
    c = np.full(55, 100.0)
    h = np.full(55, 100.05)
    l = np.full(55, 99.9)
    m200 = np.full(55, 100.0)
    l[:40] = 99.0  # 停損 99.0
    i = 40
    c[i] = 100.3
    # 同一根同時打到停損和 +1：算停損
    l[41] = 98.5
    h[41] = 101.5
    c[41] = 100.0
    tr = simulate(i, h, l, c, m200)
    assert tr["scalp_reason"] == "停損"
    assert tr["scale_reason"] == "停損"
    assert abs(tr["scalp_pts"] - (99.0 - 100.3)) < 1e-9

    # 輕踩 MA200 下方 0.15% 不會被洗掉，接著碰到 +1，之後收盤跌破 MA200 超過 0.3% 才出場
    c2 = np.full(55, 100.3)
    h2 = np.full(55, 100.4)
    l2 = np.full(55, 99.6)
    l2[:40] = 99.0
    m2 = np.full(55, 100.0)
    c2[40] = 100.3
    c2[41] = 99.85  # 跌破 0.15%，在 0.3% 緩衝內，續抱
    h2[42] = 101.4  # +1 觸價
    c2[42] = 101.1
    c2[43] = 100.6
    c2[44] = 99.60  # 跌破 MA200 0.4%，算跌回
    tr2 = simulate(40, h2, l2, c2, m2)
    assert tr2["scalp_reason"] == "+1"
    assert abs(tr2["scalp_pts"] - 1.0) < 1e-9
    assert tr2["scale_reason"] == "一半後跌回"
    # 0.5*1 + 0.5*(99.6-100.3) = 0.5 - 0.35 = 0.15
    assert abs(tr2["scale_pts"] - 0.15) < 1e-9
    assert tr2["hold"] == 2
    assert tr2["scale_hold"] == 4


if __name__ == "__main__":
    test_peel_bar_passes_and_invalid_shapes_fail()
    test_one_shot_and_rearm_after_25_under()
    test_exits_buffer_stop_and_scale()
    print("ALL TESTS PASSED")

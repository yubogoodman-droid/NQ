#!/usr/bin/env python3
"""SOXS 1m+5m+15m 撥開進場的離線測試。不打幣安。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from soxs_1m_ma200_peel import (  # noqa: E402
    entry_indices,
    htf_aligned,
    is_peel,
    last_closed_idx,
    ma60_reclaim,
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

    # MA200 末端已經上弧
    m200b = m200.copy()
    m200b[55] = 100.30
    m200b[45] = 100.0
    assert not is_peel(55, c, m7, m14, m25, m30, m99, m120, m200b)


def test_htf_alignment_and_no_lookahead() -> None:
    # 5m bars at 0, 5, 10, ... min; 15m bars at 0, 15, 30, ... min
    t5 = np.arange(300, dtype=np.int64) * 300_000
    t15 = np.arange(250, dtype=np.int64) * 900_000
    # 1m bar opening at 219*15m + 3m (closes at +4m): 219th 15m bar is NOT closed yet, must use 218
    t1_open = int(t15[219]) + 3 * 60_000
    assert last_closed_idx(t15, 900_000, t1_open) == 218
    # 1m bar opening at 219*15m + 14m (closes at +15m): 219th 15m bar JUST closed
    assert last_closed_idx(t15, 900_000, int(t15[219]) + 14 * 60_000) == 219

    m5_7 = np.full(300, 101.0)
    m5_14 = np.full(300, 100.0)
    m15_200 = np.full(250, 100.0)
    # Use a timestamp well after bar 220 of 15m (i.e. >= 3300 min)
    t5_big = np.arange(800, dtype=np.int64) * 300_000
    m5_7b = np.full(800, 101.0)
    m5_14b = np.full(800, 100.0)
    m5_99 = np.full(800, 100.0)
    m5_120 = np.full(800, 102.0)
    m5_200 = np.full(800, 101.0)
    t_ok = int(t15[220]) + 14 * 60_000
    ok, gap5, s15 = htf_aligned(t_ok, t5_big, m5_7b, m5_14b, m5_99, m5_120, m5_200, t15, m15_200)
    assert ok and gap5 is not None and gap5 > 0 and s15 == 0.0

    # 5m MA7 <= MA14 -> reject
    m5_7c = np.full(800, 99.5)
    ok2, _, _ = htf_aligned(t_ok, t5_big, m5_7c, m5_14b, m5_99, m5_120, m5_200, t15, m15_200)
    assert not ok2

    # 15m MA200 plunging (-0.5% over 10 bars) -> reject
    m15_down = np.full(250, 100.0)
    m15_down[220] = 99.5
    ok3, _, _ = htf_aligned(t_ok, t5_big, m5_7b, m5_14b, m5_99, m5_120, m5_200, t15, m15_down)
    assert not ok3

    # 5m 長均線不是 120 > 200 > 99 -> reject
    m5_120_under = np.full(800, 100.5)
    ok4, _, _ = htf_aligned(t_ok, t5_big, m5_7b, m5_14b, m5_99, m5_120_under, m5_200, t15, m15_200)
    assert not ok4


def test_ma60_reclaim_needs_a_4h_break_then_a_close_back_above() -> None:
    n = 160
    t5 = np.arange(n, dtype=np.int64) * 300_000
    c5 = np.full(n, 100.0)
    l5 = np.full(n, 99.8)
    m60 = np.full(n, 100.0)
    # bar 120 breaks the prior 48-bar low and closes under MA60
    l5[120] = 97.0
    c5[120] = 98.0
    # bar 130 is the first close back above MA60
    c5[121:130] = 99.0
    l5[121:130] = 98.8
    c5[130] = 100.4
    l5[130] = 100.1
    # decision at the close of bar 130
    t1 = int(t5[130]) + 4 * 60_000
    ok, dist = ma60_reclaim(t1, t5, l5, c5, m60, 100.0)
    assert ok and dist is not None and abs(dist) < 1e-9
    # one bar earlier the 5m close is still under MA60
    ok_early, _ = ma60_reclaim(int(t5[129]) + 4 * 60_000, t5, l5, c5, m60, 100.0)
    assert not ok_early
    # MA60 1.5% away from the 1m MA200 is too far
    ok_far, dist_far = ma60_reclaim(t1, t5, l5, c5, m60, 98.0)
    assert not ok_far and dist_far is not None and dist_far > 1.0


def test_one_shot_and_rearm_after_25_under() -> None:
    c, h, l, m7, m14, m25, m30, m99, m120, m200, t = _base_arrays(160)
    _plant_peel(c, h, l, m7, m14, m25, m30, m99, m120, m200, 55)
    _plant_peel(c, h, l, m7, m14, m25, m30, m99, m120, m200, 56)
    c[52:57] = [100.10, 100.15, 100.25, 100.40, 100.45]
    m7[54] = 99.95
    m7[55] = 100.15
    m7[56] = 100.20

    c[57:117] = 99.0
    _plant_peel(c, h, l, m7, m14, m25, m30, m99, m120, m200, 120)
    idxs = entry_indices(c, m7, m14, m25, m30, m99, m120, m200, t, t_min=0)
    assert idxs == [55, 120], idxs


def test_exits_buffer_stop_and_scale() -> None:
    c = np.full(55, 100.0)
    h = np.full(55, 100.05)
    l = np.full(55, 99.9)
    m200 = np.full(55, 100.0)
    l[:40] = 99.0
    i = 40
    c[i] = 100.3
    l[41] = 98.5
    h[41] = 101.5
    c[41] = 100.0
    tr = simulate(i, h, l, c, m200)
    assert tr["scalp_reason"] == "停損"
    assert tr["scale_reason"] == "停損"
    assert abs(tr["scalp_pts"] - (99.0 - 100.3)) < 1e-9

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
    assert abs(tr2["scale_pts"] - 0.15) < 1e-9
    assert tr2["hold"] == 2
    assert tr2["scale_hold"] == 4


if __name__ == "__main__":
    test_peel_bar_passes_and_invalid_shapes_fail()
    test_htf_alignment_and_no_lookahead()
    test_ma60_reclaim_needs_a_4h_break_then_a_close_back_above()
    test_one_shot_and_rearm_after_25_under()
    test_exits_buffer_stop_and_scale()
    print("ALL TESTS PASSED")

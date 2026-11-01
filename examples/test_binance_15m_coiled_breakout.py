from __future__ import annotations

import numpy as np

from examples.watch_binance_15m_coiled_breakout import (
    CoiledBreakoutParams,
    breakout_at,
    compute_indicators,
    detect_signals,
    follow_through_at,
)


def _make_synthetic_esports_data() -> dict:
    """構造符合底部七線黏帶 + 倍量突破 MA200 的合成 15m K 線資料。"""
    n = 260
    t0 = 1761868800000  # 2026-10-31 08:00 UTC+8
    t = np.array([t0 + k * 15 * 60_000 for k in range(n)], dtype=np.int64)

    # 前 100 根在 0.0076，之後跌到 0.0070 並長時間橫盤在 0.00710 附近讓 MA7~120 糾結，MA200 緩降到 0.00729
    c = np.full(n, 0.00710, dtype=float)
    c[:60] = 0.00780
    c[60:95] = np.linspace(0.00780, 0.00685, 35)
    c[95:235] = 0.00710
    # 230~234 微幅墊高讓 MA7 > MA14 > MA25 > MA30 > MA60/99/120
    c[230] = 0.00712
    c[231] = 0.00714
    c[232] = 0.00716
    c[233] = 0.00720
    c[234] = 0.00724
    # 第 235 根帶倍量突破 MA200 (~0.00728)
    c[235] = 0.00740
    # 第 236~238 根扇形展開續攻
    c[236] = 0.00752
    c[237] = 0.00765
    c[238:] = 0.00785

    o = c.copy()
    o[235] = 0.00724
    o[236] = 0.00740
    o[237] = 0.00752

    h = np.maximum(o, c) + 0.00002
    l = np.minimum(o, c) - 0.00002
    v = np.full(n, 100.0, dtype=float)
    v[235] = 520.0
    v[236] = 610.0
    v[237] = 480.0

    return compute_indicators({"t": t, "o": o, "h": h, "l": l, "c": c, "v": v})


def test_breakout_and_follow_through_detected():
    d = _make_synthetic_esports_data()
    params = CoiledBreakoutParams()
    sig = breakout_at(d, 235, symbol="ESPORTSUSDT", params=params)
    assert sig is not None
    assert sig.symbol == "ESPORTSUSDT"
    assert sig.breakout_idx == 235
    assert sig.coil_7_120_pct <= params.max_coil_7_120_pct
    assert sig.vol_ratio >= 4.0
    assert sig.bars_below_ma200 >= params.min_bars_below_ma200

    ft = follow_through_at(d, sig, 236, params=params)
    assert ft is not None
    assert ft.follow_idx == 236
    assert ft.follow_ext_ma200_pct >= params.follow_min_ext_ma200_pct


def test_detect_signals_rejects_low_volume_breakout():
    d = _make_synthetic_esports_data()
    d["v"][235] = 110.0  # 僅 1.1x 均量，不應觸發
    d = compute_indicators(d)
    sigs = detect_signals(d, symbol="ESPORTSUSDT")
    assert all(s.breakout_idx != 235 for s in sigs)

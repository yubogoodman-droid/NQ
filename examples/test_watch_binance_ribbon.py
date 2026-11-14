#!/usr/bin/env python3
"""Offline unit tests for 幣安 1m 黏帶三幕（優化版）。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from watch_binance_ribbon import (  # noqa: E402
    btc_above_ma25_at,
    indicators,
    kiss_at,
    lift_at,
    opt_lift_plan,
    simulate_opt_series,
    sma,
    valid_kisses,
)


def make_u_bottom_series(
    *,
    lift_upper_wick: float = 0.02,
    lift_vol_mult: float = 2.6,
    prior_high_bump: float = 2.2,
    follow_through_pct: float = 2.0,
) -> dict:
    """構造一段標準圓底吻黏帶 → 離開 → 延續的合成 1m K 線。"""
    n = 360
    t = np.arange(n, dtype=np.int64) * 60_000 + 1_790_000_000_000
    c = np.full(n, 99.85, dtype=np.float64)
    # 152..160 短暫衝高後回落，留出前 2h 高低振幅空間 (drop_120 >= 1.2%)，同時讓 MA99/120/200 收斂成黏帶
    c[162:176] = 100.28
    # 205..259 共 55 根在黏帶下方築淺圓 U 底
    u_len = 55
    xs = np.linspace(-1.0, 1.0, u_len)
    c[205:260] = 99.02 + 0.68 * (xs**2)
    # 第 260 根：微幅站上 MA200（吻）
    c[260] = 99.82
    # 261..266 貼著黏帶整理
    c[261:267] = [99.84, 99.81, 99.86, 99.90, 99.94, 99.98]
    # 第 267 根（吻後第 7 根）：放量離開
    c[267] = 100.32
    # 268..359 離開後走勢
    c[268:] = np.linspace(100.42, 100.32 * (1.0 + follow_through_pct / 100.0), n - 268)

    o = np.concatenate([[c[0]], c[:-1]])
    o[267] = 99.98
    l = np.minimum(o, c) - 0.04
    h = np.maximum(o, c) + 0.04
    h[155] = 100.0 + prior_high_bump  # 前 2h 高點影線
    # 控制離開棒上影線長度
    h[267] = c[267] + lift_upper_wick
    l[267] = 99.95

    v = np.full(n, 1000.0, dtype=np.float64)
    v[267] = 1000.0 * lift_vol_mult
    return {"t": t, "o": o, "h": h, "l": l, "c": c, "v": v}


def test_sma() -> None:
    x = np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float64)
    m = sma(x, 2)
    assert np.isnan(m[0])
    assert abs(m[1] - 3.0) < 1e-9
    assert abs(m[3] - 7.0) < 1e-9


def test_clean_u_bottom_passes_opt_filter() -> None:
    d = indicators(make_u_bottom_series())
    kisses = valid_kisses(d)
    assert 260 in kisses, f"Expected kiss at 260, got {list(kisses.keys())}"
    kiss = kisses[260]
    assert kiss["drop_120"] >= 1.2
    assert lift_at(d, 260, 267)
    plan = opt_lift_plan(d, kiss, 267, btc_ok=True)
    assert plan is not None
    assert plan["close_pos"] >= 0.70
    assert 0.50 <= plan["risk_pct"] <= 1.60
    assert plan["sl_px"] < plan["trig_px"]


def test_opt_filter_rejects_upper_wick_and_climax_and_btc_dump() -> None:
    # 1. 長上影線倒貨棒 -> 拒絕
    d_wick = indicators(make_u_bottom_series(lift_upper_wick=0.60))
    kiss_w = kiss_at(d_wick, 260)
    assert kiss_w is not None and lift_at(d_wick, 260, 267)
    assert opt_lift_plan(d_wick, kiss_w, 267, btc_ok=True) is None

    # 2. 超過 6x 天量針 -> 拒絕
    d_climax = indicators(make_u_bottom_series(lift_vol_mult=8.5))
    kiss_c = kiss_at(d_climax, 260)
    assert kiss_c is not None and lift_at(d_climax, 260, 267)
    assert opt_lift_plan(d_climax, kiss_c, 267, btc_ok=True) is None

    # 3. BTC 跌破 MA25 逆風 -> 拒絕
    d_ok = indicators(make_u_bottom_series())
    kiss_ok = kiss_at(d_ok, 260)
    assert kiss_ok is not None
    assert opt_lift_plan(d_ok, kiss_ok, 267, btc_ok=False) is None


def test_simulate_opt_series_and_btc_helper() -> None:
    raw = make_u_bottom_series(follow_through_pct=2.2)
    btc_c = np.linspace(90_000.0, 91_000.0, len(raw["t"]))
    btc_state = {
        "t": raw["t"],
        "c": btc_c,
        "m25": sma(btc_c, 25),
        "idx": {int(ts): i for i, ts in enumerate(raw["t"])},
    }
    assert btc_above_ma25_at(btc_state, int(raw["t"][267])) is True
    r30, otr = simulate_opt_series(raw, btc_state, int(raw["t"][250]), int(raw["t"][-1]))
    assert len(r30) == 1
    assert len(otr) == 1
    assert otr[0]["reason"] == "60m"
    assert otr[0]["net"] > 1.0


if __name__ == "__main__":
    test_sma()
    test_clean_u_bottom_passes_opt_filter()
    test_opt_filter_rejects_upper_wick_and_climax_and_btc_dump()
    test_simulate_opt_series_and_btc_helper()
    print("ALL TESTS PASSED")

#!/usr/bin/env python3
"""Offline unit tests for 幣安 1m 橫盤壓縮爆量起漲點監看（watch_binance_ignition.py）。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from watch_binance_ignition import (  # noqa: E402
    TZ,
    evaluate_trade_from,
    ignition_plan_at,
    indicators,
    raw_breakout_at,
    render_html_report,
    rolling_max_prev,
    rolling_min_prev,
    simulate_symbol,
    sma_prev,
    valid_ignitions,
)


def make_evaa_like_series(
    *,
    ign_vol_mult: float = 10.0,
    upper_wick_extra: float = 0.008,
    taker_buy_ratio: float = 0.67,
    prior_6h_spike: float = 0.0,
    parabolic_gain_pct: float = 45.0,
) -> dict:
    """構造一段模擬 EVAAUSDT 窄幅墊高平台 → 單根爆量收穿 → 主升浪的合成 1m K 線。"""
    n = 450
    t = np.arange(n, dtype=np.int64) * 60_000 + 1_794_640_000_000
    # 前段溫和墊高讓 MA200 走平微升、MA25 > MA99，且 2h 箱體振幅約 4.5%
    c = np.linspace(1.105, 1.148, n, dtype=np.float64)
    # 250..369 共 120 根在 1.115~1.162 窄幅橫盤
    wave = 0.012 * np.sin(np.linspace(0, 8 * np.pi, 120))
    c[250:370] = np.linspace(1.125, 1.150, 120) + wave * 0.4
    o = np.concatenate([[c[0]], c[:-1]])
    h = np.maximum(o, c) + 0.004
    l = np.minimum(o, c) - 0.004
    h[250:370] = np.minimum(h[250:370], 1.165)
    l[250:370] = np.maximum(l[250:370], 1.112)
    if prior_6h_spike > 0:
        h[120] = 1.112 * (1.0 + prior_6h_spike / 100.0)

    # 第 370 根：起漲突破棒（開 1.154 -> 收 1.199，站上 2h 箱頂 1.165）
    sig_i = 370
    o[sig_i] = 1.154
    l[sig_i] = 1.154
    c[sig_i] = 1.199
    h[sig_i] = c[sig_i] + upper_wick_extra

    # 371..435 主升段後回落觸發吊燈停利
    peak_px = h[sig_i] * (1.0 + parabolic_gain_pct / 100.0)
    ramp = np.linspace(h[sig_i] * 1.003, peak_px, 50)
    c[371:421] = ramp
    o[371:421] = ramp * 0.996
    h[371:421] = ramp * 1.005
    l[371:421] = ramp * 0.985
    # 421 根起高檔回撤 10% 觸發 -7.5% 吊燈出場
    c[421:] = peak_px * 0.89
    o[421:] = peak_px * 0.94
    h[421:] = peak_px * 0.95
    l[421:] = peak_px * 0.88

    qv = np.full(n, 22_000.0, dtype=np.float64)
    qv[sig_i] = 22_000.0 * ign_vol_mult
    tbq = qv * 0.50
    tbq[sig_i] = qv[sig_i] * taker_buy_ratio
    return {"t": t, "o": o, "h": h, "l": l, "c": c, "qv": qv, "tbq": tbq}


def test_rolling_prev_helpers_exclude_current_bar() -> None:
    x = np.array([10.0, 20.0, 15.0, 100.0], dtype=np.float64)
    mx = rolling_max_prev(x, 2)
    mn = rolling_min_prev(x, 2)
    sm = sma_prev(x, 2)
    assert np.isnan(mx[1]) and abs(mx[2] - 20.0) < 1e-9
    # index 3 的前 2 根是 [20, 15]，不含當根 100（零偷看當根）
    assert abs(mx[3] - 20.0) < 1e-9
    assert abs(mn[3] - 15.0) < 1e-9
    assert abs(sm[3] - 17.5) < 1e-9


def test_clean_evaa_like_ignition_passes_and_captures_wave() -> None:
    raw = make_evaa_like_series()
    d = indicators(raw)
    hits = valid_ignitions(d)
    assert 370 in hits, f"Expected ignition at 370, got {list(hits.keys())}"
    plan = hits[370]
    assert 2.0 <= plan["box120"] <= 6.5
    assert plan["close_pos"] >= 0.72
    assert 1.20 <= plan["risk_pct"] <= 3.00

    tr = evaluate_trade_from(d, 370, plan)
    assert tr is not None
    assert tr["fill_i"] == 371
    assert tr["hit_5_ms"] is not None and tr["hit_8_ms"] is not None
    assert tr["reason"] == "TRL"
    assert tr["net"] > 30.0


def test_filters_reject_wick_trap_climax_sell_and_wide_box() -> None:
    # 1. 長上影假突破針（如 EVAA 21:41）-> 拒絕
    d_wick = indicators(make_evaa_like_series(upper_wick_extra=0.065))
    assert raw_breakout_at(d_wick, 370) is None
    assert ignition_plan_at(d_wick, 370) is None

    # 2. 超過 55x 單根極端消息倒貨針 -> 拒絕
    d_climax = indicators(make_evaa_like_series(ign_vol_mult=80.0))
    assert raw_breakout_at(d_climax, 370) is not None
    assert ignition_plan_at(d_climax, 370) is None

    # 3. 主動買盤不足 (<56%) -> 拒絕
    d_sell = indicators(make_evaa_like_series(taker_buy_ratio=0.48))
    assert ignition_plan_at(d_sell, 370) is None

    # 4. 前 6h 已經大幅拉抬過 (>12% 振幅) -> 拒絕高位追高
    d_wide = indicators(make_evaa_like_series(prior_6h_spike=18.0))
    assert ignition_plan_at(d_wide, 370) is None


def test_cooldown_and_simulate_symbol() -> None:
    import tempfile
    from datetime import datetime

    raw = make_evaa_like_series(parabolic_gain_pct=18.0)
    with tempfile.TemporaryDirectory() as tmp:
        tdir = Path(tmp)
        trades = simulate_symbol(raw, int(raw["t"][365]), int(raw["t"][-1]), sym="EVAAUSDT", chart_dir=tdir / "img")
        assert len(trades) == 1
        assert trades[0]["sig_i"] == 370
        assert trades[0]["net"] > 8.0
        assert trades[0].get("img_name")
        out_html = render_html_report(
            trades,
            start_dt=datetime.fromtimestamp(int(raw["t"][0]) / 1000, TZ),
            end_dt=datetime.fromtimestamp(int(raw["t"][-1]) / 1000, TZ),
            days=180,
            symbol_count=368,
            out_path=tdir / "index.html",
        )
        html_txt = out_html.read_text(encoding="utf-8")
        assert "EVAAUSDT" in html_txt and "<svg" in html_txt


if __name__ == "__main__":
    test_rolling_prev_helpers_exclude_current_bar()
    test_clean_evaa_like_ignition_passes_and_captures_wave()
    test_filters_reject_wick_trap_climax_sell_and_wide_box()
    test_cooldown_and_simulate_symbol()
    print("ALL TESTS PASSED")

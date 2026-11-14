from __future__ import annotations

import numpy as np

from examples.watch_binance_15m_coiled_breakout import (
    CoiledBreakoutParams,
    breakout_at,
    compute_indicators,
    detect_signals,
    follow_through_at,
    simulate_trades,
    trade_stats,
    universe,
)


def _make_synthetic_esports_data() -> dict:
    """構造符合底部七線黏帶 + 倍量突破 MA200 的合成 15m K 線資料。"""
    n = 260
    t0 = 1761868800000  # 2026-10-31 08:00 UTC+8
    t = np.array([t0 + k * 15 * 60_000 for k in range(n)], dtype=np.int64)

    # 前段緩跌後長時間橫盤在 0.00710 附近讓 MA7~120 糾結，MA200 緩降走平至 ~0.00722（斜率約 -0.34%）
    c = np.full(n, 0.00710, dtype=float)
    c[:35] = 0.00745
    c[35:115] = 0.00740
    c[115:235] = 0.00710
    # 230~234 微幅墊高讓 MA7 > MA14 > MA25 > MA30 > MA60/99/120，且第 234 根仍在 MA200 (~0.007222) 下方
    c[230] = 0.00712
    c[231] = 0.00714
    c[232] = 0.00716
    c[233] = 0.00719
    c[234] = 0.00721
    # 第 235 根帶倍量直接從下方突破站上 MA200 (~0.007222)
    c[235] = 0.00740
    # 第 236~238 根扇形展開續攻
    c[236] = 0.00752
    c[237] = 0.00765
    c[238:] = 0.00785

    o = c.copy()
    o[233] = 0.00716
    o[234] = 0.00719
    o[235] = 0.00721
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


def test_simulate_trades_breakout_and_follow():
    d = _make_synthetic_esports_data()
    sigs = detect_signals(d, symbol="ESPORTSUSDT")
    assert len(sigs) == 1
    trades_b = simulate_trades(d, sigs, stage="breakout", trail_ma="m25", arm_r=1.2)
    trades_f = simulate_trades(d, sigs, stage="follow", trail_ma="m25", arm_r=1.2)
    assert len(trades_b) == 1
    assert len(trades_f) == 1
    assert trades_b[0].pnl_pct > 0
    assert trades_f[0].pnl_pct > 0
    st = trade_stats(trades_f)
    assert st["profit_factor"] > 1.0
    assert st["max_dd_pct"] == 0.0


def test_follow_rejects_bearish_or_lower_close_and_late_4th_bar():
    d = _make_synthetic_esports_data()
    params = CoiledBreakoutParams()
    sig = breakout_at(d, 235, symbol="ESPORTSUSDT", params=params)
    assert sig is not None

    # 確認棒若為黑 K（開高走低 c <= o），即使仍高於 MA200 +2.2% 也應拒絕
    d_bear = dict(d)
    d_bear["o"] = d["o"].copy()
    d_bear["o"][236] = 0.00760  # c[236]=0.00752 < o[236]=0.00760
    assert follow_through_at(d_bear, sig, 236, params=params) is None

    # 超過 follow_max_bars=3（第 4 根 j=239）不應觸發強確認
    assert follow_through_at(d, sig, 239, params=params) is None

    # 第二拍確認棒若帶有 >30% 長上影線（衝高被砸避雷針）應拒絕
    d_wick = dict(d)
    d_wick["h"] = d["h"].copy()
    d_wick["h"][236] = 0.00765  # o=0.00740, c=0.00752, l=0.00738 => upper wick = 0.00013 / 0.00027 = 48.1% > 30%
    assert follow_through_at(d_wick, sig, 236, params=params) is None


def test_hard_stop_and_rolling_stats():
    d = _make_synthetic_esports_data()
    sigs = detect_signals(d, symbol="ESPORTSUSDT")
    assert len(sigs) == 1
    # 模擬進場後第 2 根閃崩插針跌破 stop_price，觸價硬停損應限制在 stop_price 出場
    d_crash = dict(d)
    d_crash["l"] = d["l"].copy()
    d_crash["c"] = d["c"].copy()
    j = sigs[0].follow_idx
    assert j is not None
    d_crash["l"][j + 1] = sigs[0].stop_price * 0.90
    d_crash["c"][j + 1] = sigs[0].stop_price * 0.92
    trs = simulate_trades(d_crash, sigs, stage="follow", trail_ma="m99", arm_r=1.2, hard_stop=True)
    assert len(trs) == 1
    assert trs[0].exit_reason == "stop"
    assert abs(trs[0].exit_price - sigs[0].stop_price) < 1e-12
    st = trade_stats(trs)
    assert st["roll_taken"] == 1
    assert st["roll_1k_3x_end"] < 1000.0



def test_universe_excludes_tradfi_by_default(monkeypatch):
    import examples.watch_binance_15m_coiled_breakout as mod

    def fake_get_json(path, params=None, retries=5):
        if path == "/fapi/v1/exchangeInfo":
            return {
                "symbols": [
                    {"symbol": "BTCUSDT", "quoteAsset": "USDT", "status": "TRADING", "contractType": "PERPETUAL"},
                    {"symbol": "PLTRUSDT", "quoteAsset": "USDT", "status": "TRADING", "contractType": "TRADIFI_PERPETUAL"},
                ]
            }
        return [
            {"symbol": "BTCUSDT", "quoteVolume": "50000000"},
            {"symbol": "PLTRUSDT", "quoteVolume": "20000000"},
        ]

    monkeypatch.setattr(mod, "get_json", fake_get_json)
    assert universe(5_000_000) == ["BTCUSDT"]
    assert universe(5_000_000, include_tradfi=True) == ["BTCUSDT", "PLTRUSDT"]


#!/usr/bin/env python3
"""SOXS 假突破做空的合成測試。不打幣安。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from soxs_fake_breakout import (  # noqa: E402
    Params,
    in_session,
    run_backtest,
    score_window,
    simulate_exit,
)
from watch_soxs_big_win_tg import (  # noqa: E402
    SNIPER_PARAMS,
    detect_live_setups,
    format_setup_message,
)


def _plant_fakeout(n: int = 80, start: str = "2026-11-12 21:00", peak: float = 103.35) -> pd.DataFrame:
    """均線墊高後刺破前高，下一根收弱，再回測均線失敗。"""
    idx = pd.date_range(start, periods=n, freq="min", tz="Asia/Taipei")
    c = np.full(n, 100.0)
    c[10:36] = 102.2
    c[36:40] = 100.4
    c[40] = 101.6
    c[41] = 100.2
    c[42] = 100.7
    c[43] = 97.0
    c[44] = 98.0
    h = np.maximum(c + 0.15, c)
    l = c - 0.15
    o = np.r_[c[0], c[:-1]]
    h[28] = 103.0
    h[29:40] = np.minimum(h[29:40], 102.4)
    h[40] = peak
    l[40] = 101.2
    o[40] = 101.0
    h[41] = 101.2
    l[41] = 100.0
    o[41] = 101.4
    o[42] = 100.4
    h[42] = 102.3
    l[42] = 100.2
    o[43] = 100.6
    h[43] = 100.8
    l[43] = 96.0
    return pd.DataFrame(
        {"Open": o, "High": h, "Low": l, "Close": c, "Volume": np.full(n, 1000.0)},
        index=idx,
    )


def _params(**overrides) -> Params:
    base = dict(
        lookback=20,
        min_sep=5,
        break_window=10,
        retest_window=6,
        max_hold=8,
        min_gap_ma99=0.0,
        min_risk=0.002,
        max_risk=0.035,
        days=0,
        session="all",
        mode="limit",
    )
    base.update(overrides)
    return Params(**base)


def test_session_windows() -> None:
    hit = pd.Timestamp("2026-11-12 22:26", tz="Asia/Taipei")  # 09:26 ET
    pre_wave = pd.Timestamp("2026-11-12 22:10", tz="Asia/Taipei")  # 09:10 ET
    midday = pd.Timestamp("2026-11-12 12:00", tz="Asia/Taipei")
    early = pd.Timestamp("2026-11-12 18:00", tz="Asia/Taipei")
    friday_rth = pd.Timestamp("2026-11-13 04:30", tz="Asia/Taipei")
    saturday = pd.Timestamp("2026-11-14 22:00", tz="Asia/Taipei")
    assert in_session(hit, "tpe_us")
    assert in_session(hit, "ny")
    assert in_session(hit, "ny_open")
    assert in_session(pre_wave, "ny_open", for_wave=True)
    assert not in_session(pre_wave, "ny_open", for_wave=False)
    assert not in_session(midday, "tpe_us")
    assert not in_session(midday, "ny")
    assert not in_session(early, "tpe_us")
    assert not in_session(early, "ny")
    assert in_session(friday_rth, "tpe_us")
    assert in_session(friday_rth, "ny")
    assert in_session(friday_rth, "ny_open")
    assert not in_session(saturday, "tpe_us")
    assert not in_session(saturday, "ny")
    assert not in_session(saturday, "ny_open")
    assert in_session(saturday, "all")


def test_limit_retest_hits_target_without_future_bars() -> None:
    df = _plant_fakeout()
    trades, funnel = run_backtest(df, _params())
    assert funnel["entered"] == 1
    assert len(trades) == 1
    t = trades[0]
    assert t.entry_idx == 42
    assert t.peak_high == 103.35
    assert t.exit_reason == "target"
    assert abs(t.r_multiple - 2.5) < 1e-9
    assert t.net_pct > 0

    cut = df.iloc[:43].copy()
    early, _ = run_backtest(cut, _params())
    assert len(early) == 1
    assert early[0].entry_idx == 42
    assert early[0].entry_price == t.entry_price
    assert early[0].exit_reason == "open"

    tainted = df.copy()
    tainted.iloc[55, tainted.columns.get_loc("Low")] = 1.0
    tainted.iloc[55, tainted.columns.get_loc("High")] = 200.0
    again, _ = run_backtest(tainted, _params())
    assert again[0].entry_price == t.entry_price
    assert again[0].exit_idx == t.exit_idx
    assert again[0].exit_price == t.exit_price


def test_same_bar_stop_beats_target() -> None:
    high = np.array([101.0, 110.0, 90.0])
    low = np.array([99.0, 90.0, 90.0])
    close = np.array([100.0, 95.0, 95.0])
    idx, px, reason = simulate_exit(high, low, close, 1, 100.0, 105.0, 90.0, max_hold=2)
    assert (idx, px, reason) == (1, 105.0, "stop")


def test_runaway_overshoot_is_skipped() -> None:
    df = _plant_fakeout(peak=120.0)
    trades, funnel = run_backtest(df, _params())
    assert trades == []
    assert funnel["entered"] == 0
    assert funnel["runaway"] >= 1


def test_half_hour_score_ignores_later_target() -> None:
    df = _plant_fakeout()
    # 把 7 分鐘後的停利移到第 40 根之後，半小時結算不該算贏在停利。
    df.iloc[43, df.columns.get_loc("Low")] = 100.4
    df.iloc[43, df.columns.get_loc("Close")] = 100.5
    df.iloc[75, df.columns.get_loc("Low")] = 90.0
    df.iloc[75, df.columns.get_loc("Close")] = 91.0
    trades, _ = run_backtest(df, _params())
    assert len(trades) == 1
    early = score_window(df, trades, 30, fee_pct=0.0006)
    assert early["count"] == 1
    assert early["reasons"].get("target", 0) == 0
    later = score_window(df, trades, 60, fee_pct=0.0006)
    assert later["reasons"].get("target", 0) == 1


def test_ma99_gap_filter() -> None:
    # 拉長到 150 根以熱身 MA99：前 100 根在 100.0，拉升後掛單約 101.65，距離 MA99 (~100.5) 約 +1.1%
    idx = pd.date_range("2026-11-12 20:30", periods=150, freq="min", tz="Asia/Taipei")
    c = np.full(150, 100.0)
    c[100:126] = 102.2
    c[126:130] = 100.4
    c[130] = 101.6
    c[131] = 100.2
    c[132] = 100.7
    c[133] = 97.0
    h = c + 0.15
    l = c - 0.15
    o = np.r_[c[0], c[:-1]]
    h[118] = 103.0
    h[119:130] = np.minimum(h[119:130], 102.4)
    h[130], l[130], o[130] = 103.35, 101.2, 101.0
    h[131], l[131], o[131] = 101.2, 100.0, 101.4
    o[132], h[132], l[132] = 100.4, 102.3, 100.2
    o[133], h[133], l[133] = 100.6, 100.8, 96.0
    df = pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c, "Volume": 1000.0}, index=idx)

    ok_trades, _ = run_backtest(df, _params(min_gap_ma99=0.008))
    assert len(ok_trades) == 1
    assert ok_trades[0].gap_ma99_pct >= 0.008

    blocked, funnel = run_backtest(df, _params(min_gap_ma99=0.03))
    assert blocked == []
    assert funnel["gap_skip"] >= 1


def test_sniper_big_win_detector_arm_and_fill() -> None:
    idx = pd.date_range("2026-11-12 20:10", periods=145, freq="min", tz="Asia/Taipei")
    c = np.full(145, 48.0)
    # 21:55..22:21 拉升墊高短均，使假高 50.78 相對前 45m 低點 (~48.0) > +5.5%、離 MA60 > +2.5%
    c[105:131] = 50.15
    c[131:135] = 49.30
    c[135] = 50.10  # 22:25 刺破前高
    c[136] = 49.85  # 22:26 跌破 MA25/30 -> arm
    c[137] = 50.00  # 22:27 回測觸價 -> fill
    c[138] = 48.20
    h = c + 0.10
    l = c - 0.10
    o = np.r_[c[0], c[:-1]]
    v = np.full(145, 2000.0)
    v[110] = 15000.0  # 起漲爆量
    v[135] = 2500.0   # 創高縮量 (16.7% <= 65%)
    h[120] = 50.35    # 前高
    h[121:135] = np.minimum(h[121:135], 50.20)
    h[135], l[135], o[135] = 50.78, 49.90, 50.00
    h[136], l[136], o[136] = 50.05, 49.75, 50.05
    o[137], h[137], l[137] = 49.90, 50.25, 49.80
    df = pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c, "Volume": v}, index=idx)

    # 切在跌破剛收盤（bar 136 = 22:26 TPE = 09:26 ET）：應發出 arm 預掛單訊號
    arm_df = df.iloc[:137].copy()
    arm_hits = detect_live_setups("SOXSUSDT", arm_df, SNIPER_PARAMS, recent_bars=2)
    assert any(s.kind == "arm" for s in arm_hits)

    # 切在下一根回測觸價（bar 137 = 22:27 TPE）：應發出 fill 進場訊號
    fill_df = df.iloc[:138].copy()
    fill_hits = detect_live_setups("SOXSUSDT", fill_df, SNIPER_PARAMS, recent_bars=2)
    fills = [s for s in fill_hits if s.kind == "fill"]
    assert len(fills) == 1
    msg = format_setup_message(fill_df, fills[0])
    assert "S級大賺狙擊" in msg
    assert "SOXSUSDT" in msg

    # 若創高爆巨量（無量縮背離），大賺狙擊器必須擋掉
    df_noisy = fill_df.copy()
    df_noisy.iloc[135, df_noisy.columns.get_loc("Volume")] = 20000.0
    assert detect_live_setups("SOXSUSDT", df_noisy, SNIPER_PARAMS, recent_bars=3) == []


def test_outside_session_is_ignored() -> None:
    df = _plant_fakeout(start="2026-11-14 21:00")
    trades, funnel = run_backtest(df, _params(session="tpe_us"))
    assert trades == []
    assert funnel["entered"] == 0


if __name__ == "__main__":
    test_session_windows()
    test_limit_retest_hits_target_without_future_bars()
    test_same_bar_stop_beats_target()
    test_runaway_overshoot_is_skipped()
    test_half_hour_score_ignores_later_target()
    test_ma99_gap_filter()
    test_sniper_big_win_detector_arm_and_fill()
    test_outside_session_is_ignored()
    print("ok")

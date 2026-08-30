#!/usr/bin/env python3
"""Synthetic tests for NQ 1m 高檔 M 頭跌破 MA60 做空（不連 Yahoo）。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nq.patterns import MHeadPattern, detect_m_heads  # noqa: E402
from nq_m_head import (  # noqa: E402
    ET,
    TF_PRESETS,
    TRAIL_STEPS_5M,
    Signal,
    TradeResult,
    generate_signals,
    overlay_htf_ma60,
    overlay_m5_ma60,
    parse_period_days,
    ribbon_spread,
    ribbon_tangled,
    slow_ma_sandwich,
    far_above_ma200,
    hug_declining_ma20,
    is_swing_low_at,
    ma20_already_up,
    reclaim_stack_ok,
    untested_htf_support,
    run_backtest,
    run_tf_backtest,
    sma,
    summarize,
    write_html_report,
)


def test_parse_period_days() -> None:
    assert parse_period_days("8d") == 8
    assert parse_period_days("30d") == 30
    assert parse_period_days("1mo") == 30
    assert parse_period_days("2mo") == 60
    assert parse_period_days("4w") == 28


def test_sma() -> None:
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = sma(arr, 3)
    assert np.isnan(out[1])
    assert abs(out[2] - 2.0) < 1e-9
    assert abs(out[4] - 4.0) < 1e-9


def test_ribbon_helpers() -> None:
    assert abs(ribbon_spread(10.0, 12.0, 11.0, 13.0, 9.0) - 4.0) < 1e-9
    assert ribbon_tangled(29750.0, 29755.0, 29752.0, 29758.0, 29760.0, min_spread=28.0)
    assert not ribbon_tangled(29720.0, 29740.0, 29750.0, 29760.0, 29780.0, min_spread=28.0)
    assert ribbon_tangled(float("nan"), 1.0, 2.0, 3.0, 4.0, min_spread=1.0)


def test_summarize_short_pnl() -> None:
    class T:
        def __init__(self, pnl: float):
            self.pnl_points = pnl
            self.pnl_dollars = pnl * 20

    stats = summarize([T(40.0), T(-10.0), T(5.0)])  # type: ignore[arg-type]
    assert stats["trades"] == 3
    assert stats["wins"] == 2
    assert abs(stats["total_pnl_points"] - 35.0) < 1e-9


def _make_m_head_bars(n: int = 320) -> pd.DataFrame:
    """慢漲後末端急拉，在高檔做出夠深的雙頂，再快速跌破 MA60。"""
    close = np.zeros(n, dtype=float)
    close[0] = 20000.0
    h1, valley, h2 = 200, 218, 236

    for i in range(1, 165):
        close[i] = close[i - 1] + 0.35
    for i in range(165, h1):
        close[i] = close[i - 1] + 3.40  # 末端拉升，高峰明顯高於 MA60
    peak = close[h1 - 1] + 8.0
    close[h1] = peak
    drop = 42.0
    steps_down = valley - h1
    for i in range(h1 + 1, valley + 1):
        close[i] = peak - drop * (i - h1) / steps_down
    steps_up = h2 - valley
    for i in range(valley + 1, h2):
        close[i] = close[valley] + (peak - close[valley]) * (i - valley) / steps_up
    close[h2] = peak
    for i in range(h2 + 1, n):
        close[i] = close[i - 1] - 8.0

    high = close + 1.0
    low = close - 1.0
    peak_high = peak + 4.0
    high[h1] = peak_high
    high[h2] = peak_high
    for i in list(range(h1 - 8, h1)) + list(range(h1 + 1, h1 + 9)):
        if 0 <= i < n:
            high[i] = min(float(high[i]), peak_high - 8.0)
    for i in list(range(h2 - 8, h2)) + list(range(h2 + 1, h2 + 9)):
        if 0 <= i < n:
            high[i] = min(float(high[i]), peak_high - 8.0)
    low[valley] = close[valley] - 3.0

    idx = pd.date_range("2026-08-17 09:30", periods=n, freq="1min", tz=ET)
    return pd.DataFrame(
        {
            "open": np.r_[close[0], close[:-1]],
            "high": high,
            "low": low,
            "close": close,
            "volume": np.full(n, 80.0),
        },
        index=idx,
    )


def test_detect_m_heads_geometry() -> None:
    df = _make_m_head_bars()
    patterns = detect_m_heads(df)
    assert patterns, "synthetic M-head should be detected"
    p = patterns[0]
    assert p.first_high_idx < p.second_high_idx
    assert p.second_high <= p.first_high + 0.25
    assert abs(p.first_high - p.second_high) / p.peak < 0.002
    assert p.neckline < min(p.first_high, p.second_high)


def test_detect_and_simulate_short() -> None:
    df = _make_m_head_bars()
    funnel: dict = {}
    sigs = generate_signals(df, funnel=funnel)
    assert sigs, f"expected a short after MA60 break, funnel={funnel}"
    sig = sigs[0]
    assert sig.entry < sig.stop_loss
    assert sig.target < sig.entry
    assert sig.entry < sig.ma60 + 1e-9
    assert sig.entry < sig.pattern.neckline
    assert sig.bar_idx > sig.pattern.second_high_idx
    assert sig.ribbon_spread >= 28.0

    trades = run_backtest(df, sigs, max_bars_hold=80)
    assert trades
    assert isinstance(trades[0], TradeResult)
    # 做空：價格續跌應為正損益或至少有出場
    assert trades[0].exit_idx >= trades[0].signal.bar_idx
    assert trades[0].pnl_points == trades[0].signal.entry - trades[0].exit_price


def test_reject_higher_high() -> None:
    """右峰再創高是上漲中繼，不該當成高檔 M 頭。"""
    df = _make_m_head_bars()
    high = df["high"].to_numpy(copy=True)
    h2 = 236
    high[h2] = float(high[200]) + 8.0
    df = df.copy()
    df["high"] = high
    patterns = detect_m_heads(df)
    assert all(p.second_high_idx != h2 for p in patterns)


def test_no_signal_without_ma60_break() -> None:
    df = _make_m_head_bars()
    # 把後半段改成繼續在高檔盤整，不跌破 MA60
    close = df["close"].to_numpy(copy=True)
    h2 = 236
    hold = float(close[h2])
    for i in range(h2 + 1, len(close)):
        close[i] = hold
    df = df.copy()
    df["close"] = close
    high = df["high"].to_numpy(copy=True)
    low = df["low"].to_numpy(copy=True)
    for i in range(h2 + 1, len(close)):
        high[i] = hold + 0.8
        low[i] = hold - 0.8
    df["high"] = high
    df["low"] = low
    funnel: dict = {}
    sigs = generate_signals(df, funnel=funnel)
    assert not sigs, f"should not enter without MA60 break, got {len(sigs)} funnel={funnel}"


def test_skip_tangled_ribbon() -> None:
    """跌破結構後若均線帶寬一直打不開，整筆濾掉。"""
    df = _make_m_head_bars()
    funnel: dict = {}
    sigs = generate_signals(df, funnel=funnel, min_ribbon_spread=10_000.0)
    assert not sigs, f"impossible spread should skip, got {len(sigs)} funnel={funnel}"
    assert funnel.get("skip_tangled", 0) >= 1


def test_trail_locks_after_waterfall() -> None:
    """浮盈見過 1.6R 後鎖 1.2R；回補打到鎖利就出場，不再吐光。"""
    n = 24
    entry_idx = 6
    entry, risk = 10000.0, 100.0
    stop, target = 10100.0, 9800.0
    close = np.full(n, entry)
    high = close + 2.0
    low = close - 2.0
    # 先砸到 1.6R，再回補穿過 1.2R 鎖利
    close[entry_idx + 1] = entry - 80.0
    low[entry_idx + 1] = entry - 80.0
    high[entry_idx + 1] = entry - 60.0
    close[entry_idx + 2] = entry - 160.0
    low[entry_idx + 2] = entry - 160.0
    high[entry_idx + 2] = entry - 150.0
    close[entry_idx + 3] = entry - 110.0
    low[entry_idx + 3] = entry - 130.0
    high[entry_idx + 3] = entry - 105.0  # 穿過 1.2R=9880
    idx = pd.date_range("2026-08-25 19:00", periods=n, freq="5min", tz=ET)
    df = pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": 50.0},
        index=idx,
    )
    pattern = MHeadPattern(1, 3, 2, stop - 8.0, stop - 8.0, entry + 20.0)
    sig = Signal(
        timestamp=df.index[entry_idx],
        entry=entry,
        stop_loss=stop,
        target=target,
        pattern=pattern,
        bar_idx=entry_idx,
        ma60=entry + 30.0,
        ma20=entry + 15.0,
        ma5=entry + 5.0,
    )
    trades = run_backtest(df, [sig], max_bars_hold=12)
    assert trades
    t = trades[0]
    assert t.exit_reason == "trail_stop", t
    assert abs(t.pnl_points - 120.0) < 0.26  # 1.2R


def test_trail_does_not_block_two_r() -> None:
    """直落 2R 仍走停利，不會被鎖利搶先。"""
    n = 20
    entry_idx = 4
    entry, risk = 10000.0, 100.0
    stop, target = 10100.0, 9800.0
    close = np.full(n, entry)
    high = close + 1.0
    low = close - 1.0
    close[entry_idx + 1] = entry - 210.0
    low[entry_idx + 1] = entry - 210.0
    high[entry_idx + 1] = entry - 190.0
    idx = pd.date_range("2026-08-21 09:00", periods=n, freq="5min", tz=ET)
    df = pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": 50.0},
        index=idx,
    )
    pattern = MHeadPattern(1, 2, 1, stop - 8.0, stop - 8.0, entry + 20.0)
    sig = Signal(
        timestamp=df.index[entry_idx],
        entry=entry,
        stop_loss=stop,
        target=target,
        pattern=pattern,
        bar_idx=entry_idx,
        ma60=entry + 30.0,
        ma20=entry + 15.0,
        ma5=entry + 5.0,
    )
    trades = run_backtest(df, [sig], max_bars_hold=10)
    assert trades[0].exit_reason == "take_profit"
    assert abs(trades[0].pnl_points - 200.0) < 1e-9


def test_five_m_locks_one_r_giveback() -> None:
    """5m：浮盈見過 0.8R 後下一根鎖 0.5R，回補就出場（#2 那種）。"""
    n = 20
    entry_idx = 4
    entry, risk = 10000.0, 100.0
    stop, target = 10100.0, 9800.0
    close = np.full(n, entry)
    high = close + 2.0
    low = close - 2.0
    close[entry_idx + 1] = entry - 105.0
    low[entry_idx + 1] = entry - 105.0
    high[entry_idx + 1] = entry - 90.0  # 同一根高點仍低於 0.7R，不該出場
    close[entry_idx + 2] = entry - 50.0
    low[entry_idx + 2] = entry - 80.0
    high[entry_idx + 2] = entry - 40.0  # 穿過 0.5R=9950
    idx = pd.date_range("2026-08-05 03:30", periods=n, freq="5min", tz=ET)
    df = pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": 50.0},
        index=idx,
    )
    pattern = MHeadPattern(1, 2, 1, stop - 8.0, stop - 8.0, entry + 20.0)
    sig = Signal(
        timestamp=df.index[entry_idx],
        entry=entry,
        stop_loss=stop,
        target=target,
        pattern=pattern,
        bar_idx=entry_idx,
        ma60=entry + 30.0,
        ma20=entry + 15.0,
        ma5=entry + 5.0,
        timeframe="5m",
    )
    trades = run_backtest(df, [sig], max_bars_hold=10, trail_steps=TRAIL_STEPS_5M)
    assert trades[0].exit_reason == "trail_stop"
    assert abs(trades[0].pnl_points - 50.0) < 0.26


def test_slow_ma_sandwich() -> None:
    assert slow_ma_sandwich(29326.0, 29318.0, 29334.0)
    assert not slow_ma_sandwich(29900.0, 29874.0, 29824.0)
    assert not slow_ma_sandwich(29155.0, 29256.0, 29310.0)


def test_far_above_ma200() -> None:
    # #5：MA200 遠、1h 已破 → 擋
    assert far_above_ma200(29449.0, 29278.0, 150.0, 29550.0)
    # #6：MA200 也遠，但 1h 沒破 → 加回來
    assert not far_above_ma200(29896.75, 29730.0, 150.0, 29495.0)
    assert not far_above_ma200(29896.75, 29730.0, 150.0)  # 沒有 1h 不擋
    assert not far_above_ma200(29900.5, 29824.0, 150.0, 29069.0)  # 08-05 只高 76
    assert not far_above_ma200(29469.0, 29388.0, 150.0, 29491.0)  # 08-21 只高 81
    assert not far_above_ma200(29818.0, 29862.0, 150.0, 29599.0)  # 已破 MA200


def test_untested_htf_support() -> None:
    # #8：破 MA200、1h 還在下面 219、MA60 仍往上
    assert untested_htf_support(29818.0, 29854.0, 29849.0, 29862.0, 29599.0, 200.0)
    # #7：1h 也遠，但 MA60 已往下，放過
    assert not untested_htf_support(29873.0, 29890.0, 29895.0, 29880.0, 29536.0, 200.0)
    # 08-28：1h 只低 152，不到 200
    assert not untested_htf_support(29634.0, 29650.0, 29638.0, 29641.0, 29481.0, 200.0)
    # 還在 MA200 上方走第一條，不走這條
    assert not untested_htf_support(29896.0, 29920.0, 29910.0, 29730.0, 29495.0, 200.0)


def test_skip_slow_sandwich_still_allows_dump() -> None:
    """直落會跌出 MA120/200 夾心，5m 這條過濾不該誤殺真瀑布。"""
    df = _make_m_head_bars()
    funnel: dict = {}
    sigs = generate_signals(df, funnel=funnel, skip_slow_sandwich=True)
    assert sigs, f"clean dump should still enter, funnel={funnel}"


def test_tf_presets() -> None:
    assert TF_PRESETS["5m"]["min_bars_between_highs"] == 4
    assert TF_PRESETS["5m"]["high_level_lookback"] == 24
    assert TF_PRESETS["1m"]["swing_lookback"] == 7
    assert TF_PRESETS["1m"]["min_ribbon_spread"] == 28.0
    assert TF_PRESETS["5m"]["min_ribbon_spread"] == 28.0
    assert TF_PRESETS["1m"]["trail_steps"][0] == (1.6, 1.2)
    assert TF_PRESETS["5m"]["trail_steps"][0] == (0.8, 0.5)
    assert TF_PRESETS["5m"]["trail_steps"][1] == (1.2, 0.9)
    assert TF_PRESETS["5m"]["stop_buffer"] == 36.0
    assert TF_PRESETS["5m"]["skip_slow_sandwich"] is True
    assert TF_PRESETS["5m"]["max_above_ma200"] == 150.0
    assert TF_PRESETS["5m"]["untested_htf_gap"] == 200.0
    assert TF_PRESETS["5m"]["reclaim_exit"] is True
    assert TF_PRESETS["5m"]["reclaim_swing"] == 3
    assert TF_PRESETS["5m"]["ma20_slope_bars"] == 3
    assert "reclaim_exit" not in TF_PRESETS["1m"]
    assert "reclaim_htf" not in TF_PRESETS["5m"]
    assert "skip_before_minutes" not in TF_PRESETS["5m"]
    assert TF_PRESETS["1h"]["swing_lookback"] == 2
    assert TF_PRESETS["1h"]["stop_buffer"] == 50.0
    assert "skip_before_minutes" not in TF_PRESETS["1h"]


def test_overlay_m5_ma60() -> None:
    df = _make_m_head_bars()
    m5 = (
        df.resample("5min")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
    )
    out = overlay_m5_ma60(df, m5)
    assert "ma60_5m" in out.columns
    assert out["ma60_5m"].notna().sum() > 0
    # 未收盤的 5 分 K 不該提前出現在 1m 上：shift(1) 後開頭會是 NaN
    assert pd.isna(out["ma60_5m"].iloc[0])


def test_overlay_htf_ma60_no_peek() -> None:
    df = _make_m_head_bars()
    h1 = (
        df.resample("1h")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
    )
    out = overlay_htf_ma60(df, h1, col="ma60_1h")
    assert "ma60_1h" in out.columns
    assert pd.isna(out["ma60_1h"].iloc[0])


def test_squeeze_then_two_r() -> None:
    """0.48R 的軋空不該先鎖利；下一根砸到 2R 仍走停利。"""
    n = 20
    entry_idx = 4
    entry, risk = 10000.0, 100.0
    stop, target = 10100.0, 9800.0
    close = np.full(n, entry)
    high = close + 1.0
    low = close - 1.0
    close[entry_idx + 1] = entry - 48.0
    low[entry_idx + 1] = entry - 48.0
    high[entry_idx + 1] = entry + 16.0
    close[entry_idx + 2] = entry - 210.0
    low[entry_idx + 2] = entry - 210.0
    high[entry_idx + 2] = entry - 190.0
    idx = pd.date_range("2026-08-21 09:00", periods=n, freq="5min", tz=ET)
    df = pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": 50.0},
        index=idx,
    )
    pattern = MHeadPattern(1, 2, 1, stop - 8.0, stop - 8.0, entry + 20.0)
    sig = Signal(
        timestamp=df.index[entry_idx],
        entry=entry,
        stop_loss=stop,
        target=target,
        pattern=pattern,
        bar_idx=entry_idx,
        ma60=entry + 30.0,
        ma20=entry + 15.0,
        ma5=entry + 5.0,
        timeframe="5m",
    )
    trades = run_backtest(df, [sig], max_bars_hold=10, trail_steps=TRAIL_STEPS_5M)
    assert trades[0].exit_reason == "take_profit"
    assert abs(trades[0].pnl_points - 200.0) < 1e-9


def test_reclaim_stack_and_hug() -> None:
    assert reclaim_stack_ok(100.0, 99.0, 98.0, 97.0, 96.0)
    assert not reclaim_stack_ok(100.0, 97.0, 98.0, 97.0, 96.0)  # MA5 沒排好
    assert not reclaim_stack_ok(96.5, 99.0, 98.0, 97.0, 96.0)  # 沒收復 MA20
    assert hug_declining_ma20(100.0, 99.0, 99.8, 16.0, 0.5)  # 貼著、MA20 下彎
    assert not hug_declining_ma20(120.0, 99.0, 99.8, 16.0, 0.5)  # 離開 MA20 夠遠
    assert ma20_already_up(100.0, 99.0, 98.0)
    assert not ma20_already_up(98.0, 99.0, 100.0)
    lows = np.array([10.0, 9.0, 8.0, 7.0, 8.0, 9.0, 10.0])
    assert is_swing_low_at(lows, 3, 3)
    assert not is_swing_low_at(lows, 2, 3)


def test_breakdown_reclaim_flattens_short() -> None:
    """進場後先做出擺動低、再破那個低，然後收復+MA20 上彎 → 平空。"""
    n = 90
    close = np.full(n, 10000.0)
    high = close + 4.0
    low = close - 4.0
    # 入場後先砸出擺動低（48），彈起確認，再破底（58），再拉回
    for j, lv in enumerate((9990.0, 9980.0, 9970.0, 9960.0, 9975.0, 9990.0, 10010.0)):
        k = 45 + j
        low[k] = lv
        close[k] = lv + 8.0
        high[k] = lv + 16.0
    low[58] = 9945.0
    close[58] = 9955.0
    high[58] = 9965.0
    for j, px in enumerate((10040.0, 10080.0, 10120.0, 10150.0)):
        k = 62 + j
        close[k] = px
        low[k] = px - 8.0
        high[k] = px + 4.0
    idx = pd.date_range("2026-07-13 20:00", periods=n, freq="5min", tz=ET)
    df = pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": 50.0},
        index=idx,
    )
    pattern = MHeadPattern(1, 2, 1, 10120.0, 10120.0, 10040.0)
    sig = Signal(
        timestamp=df.index[40],
        entry=10000.0,
        stop_loss=10153.0,
        target=9694.0,
        pattern=pattern,
        bar_idx=40,
        ma60=10008.0,
        ma20=10004.0,
        ma5=10002.0,
        timeframe="5m",
    )
    trades = run_backtest(df, [sig], max_bars_hold=40, reclaim_exit=True, reclaim_swing=3)
    assert trades[0].exit_reason == "breakdown_reclaim", trades[0]
    assert trades[0].exit_idx < 80


def test_entry_cascade_bounce_is_not_reclaim() -> None:
    """進場瀑布砸完直接回補，沒有先做出擺動低再破底 → 不平。"""
    n = 80
    close = np.full(n, 10000.0)
    high = close + 4.0
    low = close - 4.0
    for j, lv in enumerate((9980.0, 9960.0, 9940.0, 9960.0, 9990.0, 10020.0, 10050.0, 10080.0)):
        k = 42 + j
        low[k] = lv
        close[k] = lv + 10.0
        high[k] = lv + 20.0
    idx = pd.date_range("2026-07-07 14:50", periods=n, freq="5min", tz=ET)
    df = pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": 50.0},
        index=idx,
    )
    pattern = MHeadPattern(1, 2, 1, 10120.0, 10120.0, 10040.0)
    sig = Signal(
        timestamp=df.index[40],
        entry=10000.0,
        stop_loss=10153.0,
        target=9694.0,
        pattern=pattern,
        bar_idx=40,
        ma60=10008.0,
        ma20=10004.0,
        ma5=10002.0,
        timeframe="5m",
    )
    trades = run_backtest(df, [sig], max_bars_hold=25, reclaim_exit=True, reclaim_swing=3)
    assert trades[0].exit_reason != "breakdown_reclaim", trades[0]


def test_swing_reclaim_without_stack_is_not_enough() -> None:
    """只翻回波段低、均線沒收回＋排列，不平。"""
    n = 80
    close = np.full(n, 10000.0)
    high = close + 2.0
    low = close - 2.0
    low[50] = 9965.0
    close[50] = 9970.0
    high[50] = 9975.0
    close[52] = 9988.0  # 翻回前低附近，但仍在均線下
    low[52] = 9975.0
    high[52] = 9990.0
    idx = pd.date_range("2026-07-13 20:00", periods=n, freq="5min", tz=ET)
    df = pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": 50.0},
        index=idx,
    )
    pattern = MHeadPattern(1, 2, 1, 10120.0, 10120.0, 10040.0)
    sig = Signal(
        timestamp=df.index[40],
        entry=10000.0,
        stop_loss=10153.0,
        target=9694.0,
        pattern=pattern,
        bar_idx=40,
        ma60=10008.0,
        ma20=10004.0,
        ma5=10002.0,
        timeframe="5m",
    )
    trades = run_backtest(df, [sig], max_bars_hold=20, reclaim_exit=True, reclaim_swing=3)
    assert trades[0].exit_reason != "breakdown_reclaim", trades[0]


def test_5m_preset_still_fires() -> None:
    df = _make_m_head_bars()
    _, trades, funnel = run_tf_backtest(df, "5m")
    assert trades, f"5m preset should still catch the synthetic dump, funnel={funnel}"
    assert trades[0].signal.timeframe == "5m"


def test_write_html_report() -> None:
    df = _make_m_head_bars()
    sigs = generate_signals(df)
    trades = run_backtest(df, sigs, max_bars_hold=80)
    out = Path("/tmp/nq_m_head_test.html")
    path = write_html_report(out, df, trades, "NQ=F", "demo")
    text = path.read_text(encoding="utf-8")
    assert "高檔M頭" in text
    if trades:
        assert "img/" in text
        img_dir = path.parent / "img"
        assert any(img_dir.glob("m01_*.png")), "expected a static trade PNG"


def main() -> int:
    test_parse_period_days()
    test_sma()
    test_ribbon_helpers()
    test_summarize_short_pnl()
    test_detect_m_heads_geometry()
    test_detect_and_simulate_short()
    test_reject_higher_high()
    test_no_signal_without_ma60_break()
    test_skip_tangled_ribbon()
    test_trail_locks_after_waterfall()
    test_trail_does_not_block_two_r()
    test_five_m_locks_one_r_giveback()
    test_slow_ma_sandwich()
    test_far_above_ma200()
    test_untested_htf_support()
    test_skip_slow_sandwich_still_allows_dump()
    test_tf_presets()
    test_overlay_m5_ma60()
    test_overlay_htf_ma60_no_peek()
    test_squeeze_then_two_r()
    test_reclaim_stack_and_hug()
    test_breakdown_reclaim_flattens_short()
    test_entry_cascade_bounce_is_not_reclaim()
    test_swing_reclaim_without_stack_is_not_enough()
    test_5m_preset_still_fires()
    test_write_html_report()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

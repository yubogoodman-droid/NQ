#!/usr/bin/env python3
"""Synthetic tests for NQ 5m 破翻回踩 MA20 (no Yahoo)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nq.ma20_retest import (  # noqa: E402
    Signal,
    TradeResult,
    choose_target_r,
    detect_kwargs,
    detect_signals,
    drop_open_end_trades,
    far_below_falling_5m_ma20,
    near_falling_5m_ma20_ma30,
    near_falling_5m_ma60,
    quality_at_entry,
    simulate,
    simulate_kwargs,
    sma,
    summarize_trades,
)
from nq_5m_ma20_retest import ET, parse_period_days, resample_ohlc, write_html_report  # noqa: E402


def test_parse_period_days() -> None:
    assert parse_period_days("8d") == 8
    assert parse_period_days("30d") == 30
    assert parse_period_days("1mo") == 30
    assert parse_period_days("4w") == 28


def test_quality_at_entry() -> None:
    assert quality_at_entry(20.0, 19.0, 18.0, 2.0)[1] == "A"
    assert quality_at_entry(20.0, 21.0, 18.0, -1.0)[1] == "B"
    assert quality_at_entry(10.0, 11.0, 12.0, -1.0)[1] == "C"


def test_near_falling_5m_ma60() -> None:
    assert near_falling_5m_ma60(29060.0, 29080.0, -8.0, 40.0)
    assert not near_falling_5m_ma60(29060.0, 29080.0, 8.0, 40.0)
    assert not near_falling_5m_ma60(29060.0, 29180.0, -8.0, 40.0)
    assert not near_falling_5m_ma60(29060.0, 29080.0, -8.0, 0.0)
    # 08-03 09:50：5m MA20 已上彎，綠線還在頭上也不算蓋頭
    assert not near_falling_5m_ma60(28470.0, 28494.8, -6.8, 40.0, 7.8)


def test_near_falling_5m_ma20_ma30() -> None:
    # 08-18 10:34 #9: below falling MA20 < MA30 < MA60, MA30 40.4 pts away
    assert near_falling_5m_ma20_ma30(
        29651.0, 29676.7, -19.9, 29691.4, -22.8, 45.0, 29715.0, -11.0
    )
    # 40-pt cap misses #9 (MA30 is 40.4)
    assert not near_falling_5m_ma20_ma30(
        29651.0, 29676.7, -19.9, 29691.4, -22.8, 40.0, 29715.0, -11.0
    )
    # 08-24 07:27: MA20/MA30 tight but MA60 below MA30 and rising — keep
    assert not near_falling_5m_ma20_ma30(
        29185.75, 29205.2, -15.7, 29212.1, -8.1, 45.0, 29197.0, 0.8
    )
    # 08-24 10:50 already above 5m MA20 — keep
    assert not near_falling_5m_ma20_ma30(
        29094.25, 29064.8, -45.8, 29115.0, -32.9, 45.0, 29164.4, -16.6
    )
    # 07-29 deep dump: MA30 47.6 pts away — keep
    assert not near_falling_5m_ma20_ma30(
        27512.25, 27535.6, -24.4, 27559.8, -63.6, 45.0, 27730.3, -50.8
    )
    # MA20 rising — keep
    assert not near_falling_5m_ma20_ma30(
        29651.0, 29676.7, 3.0, 29691.4, -22.8, 45.0, 29715.0, -11.0
    )
    # filter off
    assert not near_falling_5m_ma20_ma30(
        29651.0, 29676.7, -19.9, 29691.4, -22.8, 0.0, 29715.0, -11.0
    )


def test_far_below_falling_5m_ma20() -> None:
    # 08-28 #12：進場在下彎 5m MA20 下方 79 點，不是破底翻
    assert far_below_falling_5m_ma20(29517.25, 29596.5, -80.3, 45.0)
    # 08-28 10:27 圈：只低 23 點，要留
    assert not far_below_falling_5m_ma20(29614.25, 29637.1, -10.8, 45.0)
    # 08-18 贏的那筆低 25 點，要留
    assert not far_below_falling_5m_ma20(29577.75, 29602.6, -38.8, 45.0)
    # 08-24 在 5m MA20 上方，要留
    assert not far_below_falling_5m_ma20(29061.0, 29035.4, -35.1, 45.0)
    # 5m MA20 上彎，即使在下方也留
    assert not far_below_falling_5m_ma20(29517.25, 29596.5, 8.0, 45.0)
    assert not far_below_falling_5m_ma20(29517.25, 29596.5, -80.3, 0.0)


def test_skips_hug_falling_5m_ma60() -> None:
    df = _range_then_dump_reclaim_retest()
    open_sigs = detect_signals(df, session="day", ma60_5m_near=0.0)
    assert open_sigs
    sig = open_sigs[0]
    dist = abs(sig.entry_price - sig.ma60_5m)
    assert dist > 40.0, "08-24 style retest should sit well below 5m MA60"
    default = detect_signals(df, session="day")
    assert default, "40-pt 5m MA60 filter must keep the blue-circle style fill"
    if sig.ma60_5m_slope < 0:
        funnel: dict = {}
        blocked = detect_signals(df, session="day", ma60_5m_near=dist + 1.0, funnel=funnel)
        assert not blocked
        assert funnel.get("skip_ma60", 0) >= 1


def test_sma() -> None:
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = sma(arr, 3)
    assert np.isnan(out[1])
    assert abs(out[2] - 2.0) < 1e-9


def test_summarize_trades() -> None:
    class T:
        def __init__(self, pnl: float, quality: str = "A"):
            self.pnl_points = pnl
            self.quality = quality

    stats = summarize_trades([T(10.0, "A"), T(-4.0, "B"), T(2.0, "A")])
    assert stats["count"] == 3
    assert stats["wins"] == 2
    assert abs(stats["total_points"] - 8.0) < 1e-9


def _range_then_dump_reclaim_retest(n: int = 220) -> pd.DataFrame:
    """2h 盤整 → 破底 → 收復 MA20 → 離開 → 回踩（對齊 08-24 藍圈節奏）。

    06:30 起算：bar 90≈14:00 破底，bar 108≈15:30 回踩，落在 RTH。
    """
    close = np.full(n, 29200.0)
    high = np.full(n, 29204.0)
    low = np.full(n, 29196.0)
    for i in range(n):
        close[i] = 29200.0 + (3.0 if i % 2 == 0 else -2.0)
        high[i] = close[i] + 4.0
        low[i] = max(close[i] - 4.0, 29190.0)

    # dump → trough 28946.75
    path = [
        (90, 29120.0, 18.0, 8.0),
        (91, 29070.0, 18.0, 8.0),
        (92, 28980.0, 18.0, 8.0),
        (93, 28955.0, 18.0, 8.0),
        (94, 28950.0, 18.0, 3.25),
        (95, 28980.0, 18.0, 8.0),
        (96, 28990.0, 18.0, 8.0),
        (97, 28978.0, 18.0, 8.0),
        (98, 29012.0, 12.0, 14.0),
        (99, 29076.0, 12.0, 14.0),
        (100, 29081.0, 12.0, 14.0),
        (101, 29085.0, 12.0, 14.0),
        (102, 29106.0, 12.0, 14.0),
        (103, 29120.0, 10.0, 8.0),
        (104, 29128.0, 10.0, 8.0),
        (105, 29122.0, 10.0, 8.0),
        (106, 29110.0, 8.0, 10.0),
        (107, 29090.0, 8.0, 12.0),
        (108, 29070.0, 15.0, 8.0),  # 回踩：低點貼 MA20，不刺穿太多
    ]
    for i, px, up, dn in path:
        close[i] = px
        high[i] = px + up
        low[i] = px - dn
    low[94] = 28946.75

    for i in range(109, n):
        close[i] = close[i - 1] + 6.0
        high[i] = close[i] + 5.0
        low[i] = close[i] - 5.0

    idx = pd.date_range("2026-08-24 06:30", periods=n, freq="5min", tz=ET)
    return pd.DataFrame(
        {
            "Open": np.r_[close[0], close[:-1]],
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": np.full(n, 200.0),
        },
        index=idx,
    )


def test_detects_retest_not_reclaim() -> None:
    df = _range_then_dump_reclaim_retest()
    sigs = detect_signals(df, session="day")
    assert sigs, "expected a 回踩 MA20 signal on the synthetic dump"
    sig = sigs[0]
    assert sig.entry_idx > sig.reclaim_idx
    assert sig.reclaim_idx > sig.trough_idx
    assert sig.entry_idx - sig.reclaim_idx >= 3
    assert sig.entry_price > sig.stop_price
    # 進場應靠近 MA20，而不是收復當根遠拋
    assert abs(sig.entry_price - sig.ma20) < 80.0
    assert df["Low"].iloc[sig.entry_idx] <= sig.ma20 + 8.0 + 1e-6


def test_no_entry_if_never_leaves_ma20() -> None:
    df = _range_then_dump_reclaim_retest()
    # hug MA20 after reclaim: never put 3 bars with low > MA20+10
    sigs = detect_signals(df, session="day", leave_bars=3, leave_buffer=80.0)
    assert not sigs


def test_simulate_exits() -> None:
    df = _range_then_dump_reclaim_retest()
    sigs = detect_signals(df, session="day")
    trades = simulate(df, sigs)
    assert trades
    assert isinstance(trades[0], TradeResult)
    assert trades[0].exit_idx >= trades[0].entry_idx


def test_write_html_report(tmp_path: Path | None = None) -> None:
    df = _range_then_dump_reclaim_retest()
    sigs = detect_signals(df, session="day")
    trades = simulate(df, sigs)
    out = Path("/tmp/nq_5m_ma20_test.html") if tmp_path is None else Path(tmp_path) / "r.html"
    path = write_html_report(out, df, trades, "NQ=F", "demo")
    text = path.read_text(encoding="utf-8")
    assert "右肩在 MA20" in text
    assert "5m MA60" in text
    assert "日盤" in text
    assert "全時段" not in text
    if trades:
        assert "<img src='img/" in text
        img_dir = path.parent / "img"
        assert any(img_dir.glob("t01_*.png")), "expected a static trade PNG"


def _1m_range_dump_reclaim_retest(n: int = 420) -> pd.DataFrame:
    """1m 版：盤整 → 破底 → 收復 → 離開 15 根 → 回踩。"""
    close = np.full(n, 29200.0)
    high = np.full(n, 29204.0)
    low = np.full(n, 29196.0)
    for i in range(n):
        close[i] = 29200.0 + (3.0 if i % 2 == 0 else -2.0)
        high[i] = close[i] + 4.0
        low[i] = max(close[i] - 4.0, 29190.0)

    path = [
        (200, 29120.0, 18.0, 8.0),
        (201, 29070.0, 18.0, 8.0),
        (202, 28980.0, 18.0, 8.0),
        (203, 28955.0, 18.0, 8.0),
        (204, 28950.0, 18.0, 3.25),
        (205, 28980.0, 18.0, 8.0),
        (206, 28990.0, 18.0, 8.0),
        (207, 28978.0, 18.0, 8.0),
        (208, 29012.0, 12.0, 14.0),
        (209, 29076.0, 12.0, 14.0),
        (210, 29081.0, 12.0, 14.0),
        (211, 29085.0, 12.0, 14.0),
        (212, 29106.0, 12.0, 14.0),
    ]
    for i, px, up, dn in path:
        close[i] = px
        high[i] = px + up
        low[i] = px - dn
    low[204] = 28946.75

    for j in range(8):
        i = 213 + j
        px = 29140.0
        close[i] = px
        high[i] = px + 10.0
        low[i] = px - 4.0

    # 收復後先再往上推一波（右肩要時間），MA20 會跟上，再回踩。
    px = 29140.0
    for i in range(221, 247):
        px += 3.0
        close[i] = px
        high[i] = px + 8.0
        low[i] = px - 4.0

    ma = pd.Series(close).rolling(20, min_periods=20).mean()
    retest_i = 248
    m20 = float(ma.iloc[retest_i - 1])
    close[retest_i - 1] = m20 + 8.0
    high[retest_i - 1] = close[retest_i - 1] + 6.0
    low[retest_i - 1] = m20 + 1.0
    close[retest_i] = m20 + 2.0
    high[retest_i] = close[retest_i] + 6.0
    low[retest_i] = m20 - 4.0

    for i in range(retest_i + 1, n):
        close[i] = close[i - 1] + 1.2
        high[i] = close[i] + 4.0
        low[i] = close[i] - 4.0

    idx = pd.date_range("2026-08-24 06:30", periods=n, freq="1min", tz=ET)
    return pd.DataFrame(
        {
            "Open": np.r_[close[0], close[:-1]],
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": np.full(n, 80.0),
        },
        index=idx,
    )


def test_1m_preset_detects_retest() -> None:
    df = _1m_range_dump_reclaim_retest()
    sigs = detect_signals(df, **detect_kwargs("1m", session="day"))
    assert sigs, "expected a 1m 回踩 MA20 signal"
    sig = sigs[0]
    assert sig.entry_idx > sig.reclaim_idx
    assert sig.entry_idx - sig.reclaim_idx >= 8
    assert sig.entry_price > sig.stop_price
    assert sig.entry_price - sig.ma20 <= 8.0 + 1e-6
    trades = simulate(df, sigs, **simulate_kwargs("1m"))
    assert trades
    assert trades[0].exit_idx >= trades[0].entry_idx
    df5 = resample_ohlc(df)
    assert len(df5) > 20
    out = Path("/tmp/nq_1m_ma20_5m_compare.html")
    path = write_html_report(out, df, trades, "NQ=F", "demo", interval="1m", df_5m=df5)
    text = path.read_text(encoding="utf-8")
    assert "進場當下 5分K" in text
    assert "5m MA20" in text
    assert any((path.parent / "img").glob("t5m*.png")), "expected a 5m comparison PNG"


def test_drop_open_end_trades() -> None:
    df = _range_then_dump_reclaim_retest()
    sigs = detect_signals(df, session="day")
    trades = simulate(df, sigs)
    assert trades
    last = trades[-1]
    open_last = TradeResult(
        signal=last.signal,
        entry_idx=len(df) - 2,
        exit_idx=len(df) - 1,
        entry_price=last.entry_price,
        exit_price=last.exit_price,
        stop_price=last.stop_price,
        target_price=last.target_price,
        pnl_points=1.0,
        exit_reason="timeout",
        quality=last.quality,
    )
    kept, opened = drop_open_end_trades(df, [open_last], max_hold=36)
    assert kept == []
    assert opened == [open_last]
    kept2, opened2 = drop_open_end_trades(df, trades, max_hold=36)
    assert len(kept2) == len(trades)
    assert opened2 == []


def test_detect_kwargs_intervals() -> None:
    d1 = detect_kwargs("1m")
    d5 = detect_kwargs("5m")
    assert d1["lookback"] == 120
    assert d5["lookback"] == 24
    assert d1["leave_bars"] == 3
    assert d5["leave_bars"] == 3
    assert d1["min_break_depth"] == 10.0
    assert d1["fail_below"] == 40.0
    assert d1["ma60_5m_near"] == 40.0
    assert d5["ma60_5m_near"] == 40.0
    assert d1["max_pierce"] == 20.0
    assert d1["max_risk"] == 100.0
    assert d1["min_pullback"] == 25.0
    assert d1["max_dump_body"] == 30.0
    assert d1["dump_skip_body"] == 20.0
    assert d1["max_prev_above"] == 50.0
    assert d1["stop_at_shoulder"] is True
    assert d1["min_retest_bars"] == 8
    assert d1["max_close_above"] == 20.0
    assert d1["ma20_5m_below"] == 45.0
    assert d1["entry_until"] == 13 * 60
    assert d1["target_r_up"] == 2.0
    assert d5["stop_at_shoulder"] is False
    assert d5["max_pierce"] == 12.0
    s1 = simulate_kwargs("1m")
    assert s1["ma_exit_after"] == 60


def test_skip_gap_keeps_later_shoulder() -> None:
    """首踩被間隔擋掉時，同一波破底的晚一點右肩仍要抓到。"""
    df = _1m_range_dump_reclaim_retest()
    natural = detect_signals(df, **detect_kwargs("1m", session="day"))
    assert natural, "control: first MA20 tag should still fill with no prior trade"
    first = natural[0]
    funnel: dict = {}
    later = detect_signals(
        df,
        funnel=funnel,
        **detect_kwargs(
            "1m",
            session="day",
            min_entry_gap=5,
            min_retest_bars=0,
            min_pullback=0.0,
        ),
        last_entry_idx=first.entry_idx - 2,
    )
    assert funnel.get("skip_gap", 0) >= 1
    assert later, "later right-shoulder of the same 破底 must survive skip_gap"
    assert later[0].trough_idx == first.trough_idx
    assert later[0].entry_idx >= first.entry_idx


def test_skips_waterfall_onto_ma20() -> None:
    """07-31 09:38：大陰線從均線上方 60 點砸下來碰到 MA20，不是右肩。"""
    df0 = _1m_range_dump_reclaim_retest()
    ctrl = detect_signals(df0, **detect_kwargs("1m", session="day"))
    assert ctrl
    tag = ctrl[0].entry_idx
    df = df0.copy()
    close = df["Close"].to_numpy(copy=True)
    high = df["High"].to_numpy(copy=True)
    low = df["Low"].to_numpy(copy=True)
    opn = df["Open"].to_numpy(copy=True)
    m20 = float(pd.Series(close).rolling(20, min_periods=20).mean().iloc[tag])
    opn[tag] = m20 + 60.0
    high[tag] = opn[tag] + 3.0
    close[tag] = m20 + 4.0
    low[tag] = m20 - 4.0
    close[tag - 1] = m20 + 60.0
    high[tag - 1] = close[tag - 1] + 4.0
    low[tag - 1] = close[tag - 1] - 4.0
    # 下一根小 K 又貼 MA20：舊邏輯會買，正確是這波已作廢
    nxt = tag + 1
    close[nxt] = m20 + 3.0
    opn[nxt] = m20 + 5.0
    high[nxt] = m20 + 8.0
    low[nxt] = m20 - 3.0
    for i in range(nxt + 1, min(nxt + 8, len(close))):
        close[i] = close[i - 1] - 12.0
        high[i] = close[i] + 4.0
        low[i] = close[i] - 8.0
        opn[i] = close[i - 1]
    df = df.copy()
    df["Close"] = close
    df["High"] = high
    df["Low"] = low
    df["Open"] = opn
    funnel: dict = {}
    sigs = detect_signals(df, funnel=funnel, **detect_kwargs("1m", session="day"))
    assert funnel.get("skip_dump", 0) >= 1
    assert not any(s.entry_idx in (tag, tag + 1) for s in sigs)


def test_1m_stop_is_shoulder_not_break() -> None:
    df = _1m_range_dump_reclaim_retest()
    sigs = detect_signals(df, **detect_kwargs("1m", session="day"))
    assert sigs
    sig = sigs[0]
    assert sig.stop_price > sig.break_low - 1.0
    assert sig.entry_price - sig.stop_price < 80.0


def test_skips_close_far_above_ma20() -> None:
    """08-03 10:05：低點碰到 MA20 但收在均線上方 31 點，不是回踩；應等真正坐上均線的那根。"""
    df0 = _1m_range_dump_reclaim_retest()
    ctrl = detect_signals(df0, **detect_kwargs("1m", session="day"))
    assert ctrl
    tag = ctrl[0].entry_idx
    df = df0.copy()
    close = df["Close"].to_numpy(copy=True)
    high = df["High"].to_numpy(copy=True)
    low = df["Low"].to_numpy(copy=True)
    opn = df["Open"].to_numpy(copy=True)
    m20 = float(pd.Series(close).rolling(20, min_periods=20).mean().iloc[tag])
    # 彈走那根：影線掃 MA20，收盤卻在上方 31 點
    opn[tag] = m20 + 2.0
    low[tag] = m20 - 2.0
    close[tag] = m20 + 31.0
    high[tag] = close[tag] + 2.0
    # 下一根仍懸在均線上方，還沒回踩
    close[tag + 1] = m20 + 20.0
    opn[tag + 1] = close[tag]
    high[tag + 1] = close[tag + 1] + 4.0
    low[tag + 1] = m20 + 12.0
    # 再下一根才真正坐上 MA20（留足右肩停損空間）
    sit = tag + 2
    close[sit] = m20 + 4.0
    opn[sit] = m20 + 8.0
    high[sit] = close[sit] + 4.0
    low[sit] = m20 - 16.0
    df = df.copy()
    df["Close"] = close
    df["High"] = high
    df["Low"] = low
    df["Open"] = opn
    sigs = detect_signals(df, **detect_kwargs("1m", session="day"))
    assert sigs, "later sit on MA20 must still fill"
    assert sigs[0].entry_idx > tag, "bounce close 31 pts above MA20 is not a retest"
    assert sigs[0].entry_price - sigs[0].ma20 <= 20.0 + 1e-6


def test_1m_fills_early_right_shoulder() -> None:
    """08-28 圈在 10:30：10:27 大陰線砸上 MA20 不進，下一根小陽線才是右肩。"""
    df0 = _1m_range_dump_reclaim_retest()
    ctrl = detect_signals(df0, **detect_kwargs("1m", session="day"))
    assert ctrl
    reclaim = ctrl[0].reclaim_idx
    dump = reclaim + 9
    sit = dump + 1
    df = df0.copy()
    close = df["Close"].to_numpy(copy=True)
    high = df["High"].to_numpy(copy=True)
    low = df["Low"].to_numpy(copy=True)
    opn = df["Open"].to_numpy(copy=True)
    for t in range(reclaim + 1, dump):
        close[t] = close[t - 1] + 8.0
        high[t] = close[t] + 6.0
        low[t] = close[t] - 3.0
        opn[t] = close[t - 1]
    for _ in range(6):
        m20 = float(pd.Series(close).rolling(20, min_periods=20).mean().iloc[dump])
        close[dump - 1] = m20 + 46.0
        high[dump - 1] = close[dump - 1] + 4.0
        low[dump - 1] = m20 + 20.0
        opn[dump - 1] = close[dump - 1]
        close[dump] = m20 + 17.0
        opn[dump] = m20 + 46.0
        high[dump] = opn[dump] + 2.0
        low[dump] = m20 + 6.0
        m20_sit = float(pd.Series(close).rolling(20, min_periods=20).mean().iloc[sit])
        close[sit] = m20_sit + 12.0
        opn[sit] = m20_sit + 4.0
        high[sit] = close[sit] + 6.0
        low[sit] = m20_sit - 2.0
    for t in range(sit + 1, min(sit + 40, len(close))):
        close[t] = close[t - 1] + 4.0
        high[t] = close[t] + 4.0
        low[t] = close[t] - 3.0
        opn[t] = close[t - 1]
    df = df.copy()
    df["Close"] = close
    df["High"] = high
    df["Low"] = low
    df["Open"] = opn
    sigs = detect_signals(df, **detect_kwargs("1m", session="day", ma20_5m_below=0.0))
    assert sigs, "08-28 10:28-style right shoulder must fill after skipping the dump"
    assert sigs[0].entry_idx == sit
    assert 0.0 <= sigs[0].entry_price - sigs[0].ma20 <= 32.0 + 1e-6


def test_same_bar_wick_does_not_arm_be_stop() -> None:
    """08-28 #15：10:28 同一根先低後高，不能用這根高點把停損拉到進場再出場。"""
    idx = pd.date_range("2026-08-28 10:27", periods=8, freq="1min", tz=ET)
    entry = 29614.25
    stop = 29594.00
    risk = entry - stop
    target = entry + 1.5 * risk
    open_ = np.full(8, entry)
    high = np.full(8, entry + 4.0)
    low = np.full(8, entry + 2.0)
    close = np.full(8, entry + 3.0)
    close[0] = entry
    high[0] = 29645.75
    low[0] = 29604.00
    open_[0] = 29643.00
    # 10:28：低點仍在原始停損上方，高點剛好夠 0.8R / 近 1.2R
    open_[1] = 29613.75
    high[1] = 29638.50
    low[1] = 29611.25
    close[1] = 29627.50
    # 10:29 打到 1.5R，低點沒回到進場價
    open_[2] = 29626.00
    high[2] = 29654.00
    low[2] = 29615.75
    close[2] = 29652.75
    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": 1.0},
        index=idx,
    )
    sig = Signal(
        break_idx=0,
        trough_idx=0,
        reclaim_idx=0,
        entry_idx=0,
        entry_price=entry,
        stop_price=stop,
        target_price=target,
        break_low=29505.0,
        support=29559.0,
        ma5=entry,
        ma10=entry,
        ma20=entry,
        ma60=entry,
    )
    trades = simulate(df, [sig], max_hold=6, ma_exit_after=60, be_after_r=0.8)
    assert len(trades) == 1
    assert trades[0].exit_reason == "target"
    assert trades[0].exit_idx == 2
    assert abs(trades[0].pnl_points - 1.5 * risk) < 1e-6


def test_choose_target_r() -> None:
    assert choose_target_r(1.5, 7.8, 2.0) == 2.0
    assert choose_target_r(1.5, -10.8, 2.0) == 1.5
    assert choose_target_r(1.5, 7.8, 0.0) == 1.5
    assert choose_target_r(1.5, float("nan"), 2.0) == 1.5


def test_skips_afternoon_right_shoulder() -> None:
    """08-28 14:31 / 08-06 13:10：13:00 後的假右肩不接。"""
    df = _1m_range_dump_reclaim_retest()
    morning = detect_signals(df, **detect_kwargs("1m", session="day"))
    assert morning, "control: same setup before 13:00 must still fill"
    df = df.copy()
    # 11:00 起跳：破底約 14:24，仍在日盤，但已過 13:00
    df.index = pd.date_range("2026-08-28 11:00", periods=len(df), freq="1min", tz=ET)
    funnel: dict = {}
    late = detect_signals(df, funnel=funnel, **detect_kwargs("1m", session="day"))
    assert not late
    assert funnel.get("skip_late", 0) >= 1


def test_rth_skips_premarket_break() -> None:
    """08-12 08:30 那種夜盤破底，不能 09:30 開盤當成右肩。"""
    df = _1m_range_dump_reclaim_retest()
    # idx 204 dump → 05:06+204min = 08:30
    df = df.copy()
    df.index = pd.date_range("2026-08-24 05:06", periods=len(df), freq="1min", tz=ET)
    sigs = detect_signals(df, **detect_kwargs("1m", session="rth"))
    assert not sigs


def main() -> int:
    test_parse_period_days()
    test_quality_at_entry()
    test_near_falling_5m_ma60()
    test_near_falling_5m_ma20_ma30()
    test_far_below_falling_5m_ma20()
    test_skips_hug_falling_5m_ma60()
    test_sma()
    test_summarize_trades()
    test_detects_retest_not_reclaim()
    test_no_entry_if_never_leaves_ma20()
    test_simulate_exits()
    test_write_html_report()
    test_detect_kwargs_intervals()
    test_1m_preset_detects_retest()
    test_drop_open_end_trades()
    test_skip_gap_keeps_later_shoulder()
    test_skips_waterfall_onto_ma20()
    test_1m_stop_is_shoulder_not_break()
    test_skips_close_far_above_ma20()
    test_1m_fills_early_right_shoulder()
    test_same_bar_wick_does_not_arm_be_stop()
    test_choose_target_r()
    test_skips_afternoon_right_shoulder()
    test_rth_skips_premarket_break()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

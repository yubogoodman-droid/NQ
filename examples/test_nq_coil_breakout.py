#!/usr/bin/env python3
"""Synthetic tests for NQ 1m 均線糾結起漲點."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nq.coil import (  # noqa: E402
    ET,
    CoilSignal,
    CoilTrade,
    detect_coil_breakouts,
    make_coil_demo_bars,
    m5_asof,
    m5_asof_ma200_dist,
    m5_asof_mas,
    m5_look_at,
    quality_of,
    resample_ohlcv,
    simulate,
    sma,
    summarize_trades,
)
from nq_coil_breakout import write_html_report  # noqa: E402


def test_sma() -> None:
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = sma(arr, 3)
    assert np.isnan(out[1])
    assert abs(out[2] - 2.0) < 1e-9


def test_quality_of() -> None:
    assert quality_of(20.0, 18.0, 2.1, 14.0)[1] == "A"
    assert quality_of(40.0, 40.0, 1.5, 8.0)[1] == "C"
    assert quality_of(20.0, 18.0, 2.1, 52.0)[1] == "B"



def test_demo_catches_0735_breakout() -> None:
    df = make_coil_demo_bars()
    sigs = detect_coil_breakouts(df)
    assert sigs, "expected the 07:35-style coil breakout"
    sig = sigs[0]
    ts = df.index[sig.entry_idx]
    assert ts.hour == 7 and ts.minute >= 30
    assert sig.entry_price > sig.coil_high
    assert sig.entry_price > sig.stop_price
    assert abs(sig.stop_price - (sig.coil_low - 5.0)) < 0.01
    assert abs(sig.target_price - (sig.entry_price + 2.0 * (sig.entry_price - sig.stop_price))) < 0.01
    assert sig.vol_ratio >= 2.0
    assert sig.quality == "A"
    assert sig.ma5 > sig.ma10 > sig.ma20 > sig.ma30
    assert sig.entry_price > sig.ma200
    assert sig.ma60 < sig.ma200 and sig.ma120 < sig.ma200
    # 不該在盤整區就進場
    for s in sigs:
        assert df.index[s.entry_idx] >= pd.Timestamp("2026-08-24 07:30", tz=ET)


def test_real_chart_stands_on_ma200() -> None:
    """短 fixture 沒有日線 30 日均線，07:32 仍會進。完整回測會帶日線再過濾。"""
    path = Path(__file__).resolve().parent / "fixtures" / "nq_2026-08-24_0735.csv"
    df = pd.read_csv(path, parse_dates=["Datetime"], index_col="Datetime")
    if df.index.tz is None:
        df.index = df.index.tz_localize(ET)
    sigs = detect_coil_breakouts(df)
    hits = [s for s in sigs if df.index[s.entry_idx].strftime("%H:%M") == "07:32"]
    assert hits, f"expected 07:32 站上MA200, got {[df.index[s.entry_idx].strftime('%H:%M') for s in sigs]}"
    sig = hits[0]
    assert abs(sig.entry_price - 29217.75) < 0.3
    assert abs(sig.stop_price - (sig.coil_low - 5.0)) < 0.01
    assert abs(sig.target_price - (sig.entry_price + 2.0 * (sig.entry_price - sig.stop_price))) < 0.01
    assert sig.entry_price > sig.ma200
    assert sig.ma5 > sig.ma10 > sig.ma20 > sig.ma30
    assert sig.ma60 < sig.ma200 and sig.ma120 < sig.ma200
    late = [s for s in sigs if df.index[s.entry_idx].strftime("%H:%M") == "07:35"]
    assert not late, "07:35 is the chase bar, not the entry"
    ts = df.index[hits[0].entry_idx]
    look = m5_look_at(df, ts)
    assert look is not None
    assert look["bar_time"].strftime("%H:%M") == "07:35"
    # 07:32 當下，5分當根還沒走到 07:35 那根長綠
    assert abs(look["close"] - 29217.75) < 0.3
    assert look["forming"]
    assert abs(look["finished_close"] - 29247.25) < 1.0
    snap = m5_asof(df, ts)
    assert abs(float(snap.iloc[-1]["Close"]) - 29217.75) < 0.3
    _c5, _ma5, dist = m5_asof_ma200_dist(df)
    i = hits[0].entry_idx
    if not np.isnan(look["ma20"]):
        assert look["close"] > look["ma20"], "07:32 當時 5 分應站上 5分MA20"
        assert look["above_20"]
    if not np.isnan(look["ma30"]):
        assert look["close"] > look["ma30"], "07:32 當時 5 分應站上 5分MA30"
        assert look["above_30"]
    if not np.isnan(sig.m5_ma20):
        assert sig.m5_close > sig.m5_ma20
    if not np.isnan(sig.m5_ma30):
        assert sig.m5_close > sig.m5_ma30
    trades = simulate(df, hits)
    assert trades, "07:32 應能模擬出場"
    # 07:38 衝到 29283（>1R）後拉回，移動停利鎖 +0.3R，不再保本出場
    assert trades[0].exit_reason == "trail"
    risk = hits[0].entry_price - hits[0].stop_price
    assert abs(trades[0].pnl_points - 0.3 * risk) < 0.6


def test_real_chart_catches_1129() -> None:
    """08-24 11:21 插針後，11:29 第一次站上 MA200 要進。"""
    path = Path(__file__).resolve().parent / "fixtures" / "nq_2026-08-24_1129.csv"
    df = pd.read_csv(path, parse_dates=["Datetime"], index_col="Datetime")
    if df.index.tz is None:
        df.index = df.index.tz_localize(ET)
    sigs = detect_coil_breakouts(df)
    hits = [s for s in sigs if df.index[s.entry_idx].strftime("%H:%M") == "11:29"]
    assert hits, f"expected 11:29 站上MA200, got {[df.index[s.entry_idx].strftime('%H:%M') for s in sigs]}"
    sig = hits[0]
    assert abs(sig.entry_price - 29121.25) < 0.3
    assert sig.entry_price > sig.ma200
    assert sig.ma5 > sig.ma10 > sig.ma20 > sig.ma30
    assert sig.ma60 < sig.ma200 and sig.ma120 < sig.ma200
    assert sig.vol_ratio >= 2.0
    assert sig.body <= 40.0
    look = m5_look_at(df, df.index[sig.entry_idx])
    assert look is not None
    if not np.isnan(look["ma30"]):
        assert look["close"] > look["ma30"]


def test_trail_locks_after_1r() -> None:
    idx = pd.date_range("2026-08-24 07:32", periods=6, freq="1min", tz=ET)
    close = np.array([100.0, 108.0, 112.0, 104.0, 103.0, 102.0])
    high = np.array([100.0, 111.0, 113.0, 112.0, 104.0, 103.0])
    low = np.array([99.0, 107.0, 110.0, 102.5, 102.0, 101.0])
    df = pd.DataFrame(
        {
            "Open": close,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": np.full(6, 100.0),
        },
        index=idx,
    )
    sig = CoilSignal(
        coil_start_idx=0,
        coil_end_idx=0,
        entry_idx=0,
        entry_price=100.0,
        stop_price=90.0,
        target_price=120.0,
        coil_high=95.0,
        coil_low=91.0,
        coil_range=4.0,
        ribbon_width=5.0,
        vol_ratio=3.0,
        prior_drop=40.0,
        body=8.0,
        ma5=101.0,
        ma10=100.0,
        ma20=99.0,
        ma30=98.0,
        ma60=97.0,
        ma100=96.0,
        ma120=95.0,
        ma200=94.0,
    )
    trades = simulate(df, [sig])
    assert trades[0].exit_reason == "trail"
    assert abs(trades[0].pnl_points - 3.0) < 1e-9


def test_m5_asof_ma200_dist_matches_look_at() -> None:
    n = 200 * 5 + 8
    close = np.full(n, 30000.0, dtype=float)
    close[-4:] = 29880.0
    idx = pd.date_range("2026-07-01 00:00", periods=n, freq="1min", tz=ET)
    df = pd.DataFrame(
        {
            "Open": np.r_[close[0], close[:-1]],
            "High": close + 2.0,
            "Low": close - 2.0,
            "Close": close,
            "Volume": np.full(n, 80.0),
        },
        index=idx,
    )
    _c5, ma20, ma30, _ma200, dist = m5_asof_mas(df)
    look = m5_look_at(df, df.index[-1])
    assert look is not None
    assert not np.isnan(dist[-1])
    assert not np.isnan(look["ma200"])
    assert abs(dist[-1] - (look["close"] - look["ma200"])) < 1.5
    assert dist[-1] < -100
    assert not np.isnan(ma20[-1])
    assert abs(ma20[-1] - look["ma20"]) < 1.5
    assert not np.isnan(ma30[-1])
    assert abs(ma30[-1] - look["ma30"]) < 1.5


def test_skip_5m_waterfall_bounce() -> None:
    """5 分還在 MA200 下方很深時，若打開 5 分 MA200 過濾，1 分糾結突破當大空反彈，不接。"""
    head_n = 1000
    head_close = np.array([29450.0 + (3.0 if i % 2 == 0 else -3.0) for i in range(head_n)])
    head_idx = pd.date_range("2026-08-23 09:20", periods=head_n, freq="1min", tz=ET)
    head = pd.DataFrame(
        {
            "Open": np.r_[head_close[0], head_close[:-1]],
            "High": head_close + 2.0,
            "Low": head_close - 2.0,
            "Close": head_close,
            "Volume": np.full(head_n, 90.0),
        },
        index=head_idx,
    )
    tail = make_coil_demo_bars()
    df = pd.concat([head, tail])
    df = df[~df.index.duplicated(keep="last")]
    with_filter = detect_coil_breakouts(
        df, max_m5_below_200=0.0, require_m5_above_ma30=False
    )
    without = detect_coil_breakouts(
        df, max_m5_below_200=-1.0, require_m5_above_ma30=False
    )
    assert without, "關掉 5 分深度過濾後，模擬圖仍應有起漲點"
    if with_filter:
        for s in with_filter:
            assert np.isnan(s.m5_dist) or s.m5_dist > 0.0


def test_require_above_5m_ma30() -> None:
    df = make_coil_demo_bars()
    allowed = detect_coil_breakouts(df)
    assert allowed, "模擬圖 5 分應站上 MA30"
    sig = allowed[0]
    if not np.isnan(sig.m5_ma30):
        assert sig.m5_close > sig.m5_ma30
    head_n = 250
    head_close = np.full(head_n, 29450.0)
    head_idx = pd.date_range("2026-08-23 21:50", periods=head_n, freq="1min", tz=ET)
    head = pd.DataFrame(
        {
            "Open": np.r_[head_close[0], head_close[:-1]],
            "High": head_close + 2.0,
            "Low": head_close - 2.0,
            "Close": head_close,
            "Volume": np.full(head_n, 90.0),
        },
        index=head_idx,
    )
    high = pd.concat([head, df])
    high = high[~high.index.duplicated(keep="last")]
    blocked = detect_coil_breakouts(high)
    if blocked:
        for s in blocked:
            assert np.isnan(s.m5_ma30) or s.m5_close > s.m5_ma30
    off = detect_coil_breakouts(
        high, require_m5_above_ma30=False, require_m5_above_ma20=False
    )
    assert off, "關掉 5 分 MA20/MA30 後模擬圖仍應有起漲點"


def test_skip_chase_body() -> None:
    path = Path(__file__).resolve().parent / "fixtures" / "nq_2026-08-24_0735.csv"
    df = pd.read_csv(path, parse_dates=["Datetime"], index_col="Datetime")
    if df.index.tz is None:
        df.index = df.index.tz_localize(ET)
    sigs = detect_coil_breakouts(df, max_body=5.0)
    hits = [s for s in sigs if df.index[s.entry_idx].strftime("%H:%M") == "07:32"]
    assert not hits, "實體 7.8 應被 max_body=5 擋掉"


def test_failed_breakout_exits_on_two_closes() -> None:
    idx = pd.date_range("2026-08-24 07:32", periods=8, freq="1min", tz=ET)
    close = np.array([110.0, 108.0, 99.0, 98.0, 97.0, 96.0, 95.0, 94.0])
    df = pd.DataFrame(
        {
            "Open": close,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": np.full(8, 100.0),
        },
        index=idx,
    )
    sig = CoilSignal(
        coil_start_idx=0,
        coil_end_idx=0,
        entry_idx=0,
        entry_price=110.0,
        stop_price=80.0,
        target_price=170.0,
        coil_high=100.0,
        coil_low=85.0,
        coil_range=15.0,
        ribbon_width=10.0,
        vol_ratio=3.0,
        prior_drop=40.0,
        body=8.0,
        ma5=109.0,
        ma10=108.0,
        ma20=107.0,
        ma30=106.0,
        ma60=105.0,
        ma100=104.0,
        ma120=103.0,
        ma200=102.0,
    )
    trades = simulate(df, [sig])
    assert trades
    assert trades[0].exit_reason == "fail"
    assert trades[0].exit_idx == 3
    assert abs(trades[0].exit_price - 98.0) < 1e-9
    assert trades[0].pnl_points < 0


def test_no_signal_in_wide_trend() -> None:
    n = 320
    close = np.linspace(29000.0, 29600.0, n)
    idx = pd.date_range("2026-08-24 02:00", periods=n, freq="1min", tz=ET)
    df = pd.DataFrame(
        {
            "Open": np.r_[close[0], close[:-1]],
            "High": close + 4.0,
            "Low": close - 4.0,
            "Close": close,
            "Volume": np.full(n, 200.0),
        },
        index=idx,
    )
    sigs = detect_coil_breakouts(df)
    assert not sigs, "a one-way trend with fanned MAs should not count as 起漲"


def test_simulate_and_html(tmp_path: Path | None = None) -> None:
    df = make_coil_demo_bars()
    sigs = detect_coil_breakouts(df)
    trades = simulate(df, sigs)
    assert trades
    assert isinstance(trades[0], CoilTrade)
    stats = summarize_trades(trades)
    assert stats["count"] == len(trades)
    out = Path("/tmp/nq_coil_test.html") if tmp_path is None else Path(tmp_path) / "r.html"
    path = write_html_report(out, df, trades, "NQ=F", "demo")
    text = path.read_text(encoding="utf-8")
    assert "起漲點" in text
    assert "<img src='img/" in text
    assert "5分K 當時" in text
    img_dir = path.parent / "img"
    assert any(img_dir.glob("m1_t01_*.png")), "expected a static 1m trade PNG"
    assert any(img_dir.glob("m5_t01_*.png")), "expected the 5m as-of snapshot"


def test_resample_ohlcv_5m() -> None:
    df = make_coil_demo_bars()
    m5 = resample_ohlcv(df, "5min")
    assert not m5.empty
    assert list(m5.columns)[:4] == ["Open", "High", "Low", "Close"]
    assert abs(float(m5["High"].max()) - float(df["High"].max())) < 1e-9
    assert abs(float(m5["Low"].min()) - float(df["Low"].min())) < 1e-9
    ts = m5.index[12]
    win = df.loc[(df.index > ts - pd.Timedelta(minutes=5)) & (df.index <= ts)]
    assert len(win) >= 1
    assert abs(float(m5.loc[ts, "Close"]) - float(win["Close"].iloc[-1])) < 1e-9
    assert abs(float(m5.loc[ts, "High"]) - float(win["High"].max())) < 1e-9
    assert abs(float(m5.loc[ts, "Low"]) - float(win["Low"].min())) < 1e-9


def test_html_1m_5m_asof(tmp_path: Path | None = None) -> None:
    df1 = make_coil_demo_bars()
    trades1 = simulate(df1, detect_coil_breakouts(df1))
    out = Path("/tmp/nq_coil_compare.html") if tmp_path is None else Path(tmp_path) / "c.html"
    path = write_html_report(out, df1, trades1, "NQ=F", "demo")
    text = path.read_text(encoding="utf-8")
    assert "5分K 當時" in text
    assert "未收完" in text or "已收完" in text
    img_dir = path.parent / "img"
    assert any(img_dir.glob("m1_t01_*.png"))
    assert any(img_dir.glob("m5_t01_*.png"))


def main() -> int:
    test_sma()
    test_quality_of()
    test_demo_catches_0735_breakout()
    test_real_chart_stands_on_ma200()
    test_real_chart_catches_1129()
    test_trail_locks_after_1r()
    test_m5_asof_ma200_dist_matches_look_at()
    test_skip_5m_waterfall_bounce()
    test_require_above_5m_ma30()
    test_skip_chase_body()
    test_failed_breakout_exits_on_two_closes()
    test_no_signal_in_wide_trend()
    test_simulate_and_html()
    test_resample_ohlcv_5m()
    test_html_1m_5m_asof()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
    CoilTrade,
    detect_coil_breakouts,
    make_coil_demo_bars,
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


def test_demo_catches_0735_breakout() -> None:
    df = make_coil_demo_bars()
    sigs = detect_coil_breakouts(df)
    assert sigs, "expected the 07:35-style coil breakout"
    sig = sigs[0]
    ts = df.index[sig.entry_idx]
    assert ts.hour == 7 and ts.minute >= 30
    assert sig.entry_price > sig.coil_high
    assert sig.entry_price > sig.stop_price
    assert sig.vol_ratio >= 2.0
    assert sig.quality == "A"
    assert sig.ma5 > sig.ma10 > sig.ma20 > sig.ma30
    assert sig.entry_price > sig.ma200
    assert sig.ma60 < sig.ma200 and sig.ma120 < sig.ma200
    # 不該在盤整區就進場
    for s in sigs:
        assert df.index[s.entry_idx] >= pd.Timestamp("2026-08-24 07:30", tz=ET)


def test_real_chart_stands_on_ma200() -> None:
    """你貼的那張圖：進場是 07:32 站上 MA200（29217.75），不是 07:35 放量追高。"""
    path = Path(__file__).resolve().parent / "fixtures" / "nq_2026-08-24_0735.csv"
    df = pd.read_csv(path, parse_dates=["Datetime"], index_col="Datetime")
    if df.index.tz is None:
        df.index = df.index.tz_localize(ET)
    sigs = detect_coil_breakouts(df)
    hits = [s for s in sigs if df.index[s.entry_idx].strftime("%H:%M") == "07:32"]
    assert hits, f"expected 07:32 站上MA200, got {[df.index[s.entry_idx].strftime('%H:%M') for s in sigs]}"
    sig = hits[0]
    assert abs(sig.entry_price - 29217.75) < 0.3
    assert sig.entry_price > sig.ma200
    assert sig.ma5 > sig.ma10 > sig.ma20 > sig.ma30
    assert sig.ma60 < sig.ma200 and sig.ma120 < sig.ma200
    late = [s for s in sigs if df.index[s.entry_idx].strftime("%H:%M") == "07:35"]
    assert not late, "07:35 is the chase bar, not the entry"
    m5 = resample_ohlcv(df, "5min")
    ts = df.index[hits[0].entry_idx]
    pos = int(m5.index.searchsorted(ts))
    assert m5.index[pos].strftime("%H:%M") == "07:35"
    assert abs(float(m5.iloc[pos]["Close"]) - 29247.25) < 1.0


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
    img_dir = path.parent / "img"
    assert any(img_dir.glob("*t01_*.png")), "expected a static trade PNG"


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


def test_same_rules_on_5m_bars() -> None:
    """同一套 K 數規則套在 5 分序列上仍抓得到那波起漲。"""
    df = make_coil_demo_bars()
    df = df.copy()
    df.index = pd.date_range("2026-08-24 02:00", periods=len(df), freq="5min", tz=ET)
    sigs = detect_coil_breakouts(df)
    assert sigs
    sig = sigs[0]
    assert sig.entry_price > sig.coil_high
    assert sig.ma5 > sig.ma10 > sig.ma20 > sig.ma30
    assert sig.entry_price > sig.ma200
    assert sig.ma60 < sig.ma200 and sig.ma120 < sig.ma200


def test_html_1m_5m_compare(tmp_path: Path | None = None) -> None:
    df1 = make_coil_demo_bars()
    trades1 = simulate(df1, detect_coil_breakouts(df1))
    df5 = df1.copy()
    df5.index = pd.date_range("2026-08-24 02:00", periods=len(df5), freq="5min", tz=ET)
    trades5 = simulate(df5, detect_coil_breakouts(df5))
    out = Path("/tmp/nq_coil_compare.html") if tmp_path is None else Path(tmp_path) / "c.html"
    path = write_html_report(
        out,
        df1,
        trades1,
        "NQ=F",
        "demo",
        interval="1m",
        others=[("5m", df5, trades5, {})],
    )
    text = path.read_text(encoding="utf-8")
    assert "1分K" in text and "5分K" in text
    assert "同日進場對照" in text
    assert "1分進場" in text and "5分當根" in text
    img_dir = path.parent / "img"
    assert any(img_dir.glob("m1_t01_*.png"))
    assert any(img_dir.glob("m5_t01_*.png"))


def main() -> int:
    test_sma()
    test_quality_of()
    test_demo_catches_0735_breakout()
    test_real_chart_stands_on_ma200()
    test_no_signal_in_wide_trend()
    test_simulate_and_html()
    test_resample_ohlcv_5m()
    test_same_rules_on_5m_bars()
    test_html_1m_5m_compare()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

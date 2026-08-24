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


def test_real_chart_0735() -> None:
    """你貼的那張圖：08-24 07:07 低點 29145.75，07:35 放量長綠起漲。"""
    path = Path(__file__).resolve().parent / "fixtures" / "nq_2026-08-24_0735.csv"
    df = pd.read_csv(path, parse_dates=["Datetime"], index_col="Datetime")
    if df.index.tz is None:
        df.index = df.index.tz_localize(ET)
    sigs = detect_coil_breakouts(df)
    hits = [s for s in sigs if df.index[s.entry_idx].strftime("%H:%M") == "07:35"]
    assert hits, f"expected 07:35 起漲, got {[df.index[s.entry_idx].strftime('%H:%M') for s in sigs]}"
    sig = hits[0]
    assert abs(sig.entry_price - 29247.25) < 0.3
    assert sig.coil_high < 29220
    assert sig.coil_low > 29160
    assert sig.vol_ratio >= 2.0
    assert sig.body >= 10.0
    assert sig.ma5 > sig.ma10 > sig.ma20 > sig.ma30
    assert sig.entry_price > sig.ma200
    assert sig.ma60 < sig.ma200 and sig.ma120 < sig.ma200


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
    assert any(img_dir.glob("t01_*.png")), "expected a static trade PNG"


def main() -> int:
    test_sma()
    test_quality_of()
    test_demo_catches_0735_breakout()
    test_real_chart_0735()
    test_no_signal_in_wide_trend()
    test_simulate_and_html()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

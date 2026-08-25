#!/usr/bin/env python3
"""Synthetic tests for NQ 1m 站上季線（no Yahoo）."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nq_ma60_stand import (  # noqa: E402
    ET,
    TradeResult,
    detect_signals,
    m5_context,
    macd,
    quality_from_stand,
    resample_m5,
    simulate,
    write_html_report,
)
from nq_ma_reclaim import sma  # noqa: E402


def test_quality_from_stand() -> None:
    assert quality_from_stand(0.0, 1.6, 2.0, 1.0, 10.0, 12.0, 0.4) == (3, "A")
    assert quality_from_stand(-10.0, 0.8, 2.0, 1.0, 40.0, 40.0, 0.4) == (1, "B")
    assert quality_from_stand(-10.0, 0.8, 0.5, 1.0, 40.0, 40.0, -0.2) == (0, "C")


def test_macd_cross() -> None:
    close = np.concatenate([np.linspace(100, 80, 40), np.linspace(80, 120, 40)])
    dif, dea, hist = macd(close)
    assert len(dif) == len(close)
    assert not np.isnan(dif[-1])
    assert dif[-1] > dea[-1]
    assert hist[-1] > 0


def _make_stand_bars(n: int = 220) -> pd.DataFrame:
    """Grind below a rising path, stay under MA60, then a bullish stand-up bar."""
    close = np.zeros(n, dtype=float)
    close[0] = 20000.0
    for i in range(1, 80):
        close[i] = close[i - 1] + 0.35
    for i in range(80, 180):
        # chop just under the eventual MA60
        close[i] = close[i - 1] + (0.15 if i % 3 else -0.25)
    # last stretch: stay clearly below a flattening MA, then pop through
    base = close[179]
    for i in range(180, 210):
        close[i] = base - 8.0 + (1.2 if i % 2 == 0 else -0.8)
    close[210] = base + 18.0
    for i in range(211, n):
        close[i] = close[i - 1] + 1.4
    high = close + 1.2
    low = close - 1.2
    low[210] = close[210] - 4.0
    high[210] = close[210] + 2.0
    vol = np.full(n, 50.0)
    vol[210] = 180.0
    idx = pd.date_range("2026-08-17 20:00", periods=n, freq="1min", tz=ET)
    return pd.DataFrame(
        {
            "Open": np.r_[close[0], close[:-1]],
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": vol,
        },
        index=idx,
    )


def test_detect_stand_on_ma60() -> None:
    df = _make_stand_bars()
    sigs = detect_signals(df, skip_hour_start=None, skip_hour_end=None, use_cluster=False)
    assert sigs, "expected a 站上季線 signal on the synthetic pop"
    sig = sigs[0]
    assert sig.entry_price > sig.ma60
    assert sig.entry_price > sig.stop_price
    assert sig.below_bars >= 8
    assert sig.quality in {"A", "B", "C"}
    close = df["Close"].to_numpy(float)
    ma60 = sma(close, 60)
    i = sig.entry_idx
    assert close[i - 1] <= ma60[i - 1]
    assert close[i] > ma60[i]


def test_skip_bearish_cross() -> None:
    df = _make_stand_bars()
    i = 210
    df.iloc[i, df.columns.get_loc("Open")] = float(df["Close"].iloc[i]) + 6.0
    sigs = detect_signals(df, skip_hour_start=None, skip_hour_end=None, use_cluster=False)
    assert not any(s.entry_idx == i for s in sigs)


def test_simulate_target_or_stop() -> None:
    df = _make_stand_bars()
    sigs = detect_signals(df, skip_hour_start=None, skip_hour_end=None, use_cluster=False)
    trades = simulate(df, sigs, preopen_flat=False)
    assert trades
    assert isinstance(trades[0], TradeResult)
    assert trades[0].exit_idx >= trades[0].entry_idx
    assert trades[0].exit_reason in {"target", "stop", "timeout", "preopen_flat"}


def test_resample_m5() -> None:
    df = _make_stand_bars()
    m5 = resample_m5(df)
    assert len(m5) >= 40
    assert {"Open", "High", "Low", "Close", "Volume"} <= set(m5.columns)
    ctx = m5_context(m5, df.index[210])
    assert ctx["idx"] >= 0
    assert ctx["close"] > 0


def test_write_html_report(tmp_path: Path | None = None) -> None:
    df = _make_stand_bars()
    sigs = detect_signals(df, skip_hour_start=None, skip_hour_end=None, use_cluster=False)
    trades = simulate(df, sigs, preopen_flat=False)
    out = Path("/tmp/nq_ma60_stand_test.html") if tmp_path is None else Path(tmp_path) / "r.html"
    path = write_html_report(out, df, trades, "NQ=F", "demo")
    text = path.read_text(encoding="utf-8")
    assert "站上季線" in text
    assert "五分K對照" in text
    if trades:
        assert "<img src='img/" in text
        img_dir = path.parent / "img"
        assert any(img_dir.glob("t01_*.png")), "expected a static trade PNG"
        assert any(img_dir.glob("t01_*_5m.png")), "expected a 5m comparison PNG"


def main() -> int:
    test_quality_from_stand()
    test_macd_cross()
    test_detect_stand_on_ma60()
    test_skip_bearish_cross()
    test_simulate_target_or_stop()
    test_resample_m5()
    test_write_html_report()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

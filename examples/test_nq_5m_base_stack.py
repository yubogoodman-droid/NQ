#!/usr/bin/env python3
"""Synthetic tests for NQ 5m 打底後 5/10/20 多排 (no Yahoo)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nq_5m_base_stack import (  # noqa: E402
    ET,
    LOOSE_DETECT,
    TradeResult,
    _ref_window,
    _zoom_window,
    detect_signals,
    quality_from_setup,
    simulate,
    sma,
    summarize_trades,
    write_html_report,
)


def test_quality_from_setup() -> None:
    assert quality_from_setup(152.0, 0.43, 16.8, 23.9) == (3, "A")
    assert quality_from_setup(152.0, 0.43, 16.8, 0.0) == (2, "A")
    assert quality_from_setup(90.0, 0.43, 40.0) == (1, "B")
    assert quality_from_setup(90.0, 0.80, 40.0) == (0, "C")


def test_sma() -> None:
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = sma(arr, 3)
    assert np.isnan(out[1])
    assert abs(out[2] - 2.0) < 1e-9
    assert abs(out[4] - 4.0) < 1e-9


def test_summarize_trades() -> None:
    class T:
        def __init__(self, pnl: float, quality: str = "A"):
            self.pnl_points = pnl
            self.quality = quality

    stats = summarize_trades([T(10.0, "A"), T(-4.0, "B"), T(2.0, "A")])
    assert stats["count"] == 3
    assert stats["wins"] == 2
    assert abs(stats["total_points"] - 8.0) < 1e-9
    assert stats["by_quality"]["A"]["n"] == 2


def _make_base_stack_bars(
    n: int = 220, bounce: bool = True, *, slow: bool = False, knot: bool = False, deep: bool = False
) -> pd.DataFrame:
    """Grind, dump, optional short base then lift until 5/10/20 fan."""
    close = np.zeros(n, dtype=float)
    close[0] = 20000.0
    for i in range(1, 90):
        close[i] = close[i - 1] + (0.35 if i % 2 == 0 else -0.15)
    peak = close[89]
    dump_step = 16.0 if deep else 8.5
    lift_after = 8.0 if deep else 5.0
    if slow:
        for k, i in enumerate(range(90, 130)):
            close[i] = peak - 3.2 * (k + 1)
        floor = close[129]
        low_i = 132
        close[low_i] = floor - 8.0
        bottom = close[low_i]
        for i in range(130, low_i):
            close[i] = floor + (2.0 if i % 2 == 0 else -1.0)
        for i in range(low_i + 1, n):
            close[i] = close[i - 1] + (0.8 if bounce else -3.0)
    else:
        for k, i in enumerate(range(90, 106)):
            close[i] = peak - dump_step * (k + 1)
        floor = close[105]
        low_i = 108
        close[low_i] = floor - 8.0
        bottom = close[low_i]
        for i in range(106, low_i):
            close[i] = floor + (2.0 if i % 2 == 0 else -1.0)
        if bounce and not knot:
            bounce_step = 9.0 if deep else 3.5
            bounce_base = 15.0 if deep else 8.0
            for i in range(low_i + 1, 118):
                close[i] = bottom + bounce_base + bounce_step * (i - low_i)
            for i in range(118, n):
                close[i] = min(peak + 20.0, close[i - 1] + lift_after)
        elif bounce and knot:
            for i in range(low_i + 1, n):
                close[i] = bottom + 10.0 + (1.2 if i % 2 == 0 else 0.4)
        else:
            for i in range(low_i + 1, n):
                close[i] = close[i - 1] - 5.0

    high = close + 1.8
    low = close - 1.8
    open_ = np.r_[close[0], close[:-1]]
    dump_end = 130 if slow else 106
    dump_start = 90
    for i in range(dump_start, dump_end):
        open_[i] = close[i] + 4.0
        high[i] = open_[i] + 0.5
        low[i] = close[i] - 2.2
    low[low_i] = close[low_i] - 1.0
    high[low_i] = close[low_i] + 2.0
    if bounce:
        for i in range(low_i + 1, min(low_i + 12, n)):
            open_[i] = close[i] - 1.2
            low[i] = min(close[i] - 1.2, bottom + 3.0)
            high[i] = close[i] + 1.5

    vol = np.full(n, 90.0)
    vol[dump_start:low_i + 1] = 200.0
    idx = pd.date_range("2026-08-25 18:00", periods=n, freq="5min", tz=ET)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=idx,
    )


def test_detect_base_then_stack() -> None:
    df = _make_base_stack_bars(bounce=True)
    sigs = detect_signals(df)
    assert sigs, "expected 5/10/20 stack after the synthetic base"
    sig = sigs[0]
    assert sig.entry_idx > sig.base_idx
    assert sig.entry_idx - sig.base_idx >= 6
    assert sig.entry_price > sig.stop_price
    assert sig.drop_pts >= 80
    assert sig.ma5 > sig.ma10 > sig.ma20
    assert sig.entry_price > sig.ma5
    risk = sig.entry_price - sig.stop_price
    assert (sig.target_price - sig.entry_price) / risk >= 1.99


def test_no_signal_on_continued_dump() -> None:
    df = _make_base_stack_bars(bounce=False)
    sigs = detect_signals(df)
    assert not sigs, "waterfall after the low should not count as 打底多排"


def test_simulate_and_html(tmp_path: Path | None = None) -> None:
    df = _make_base_stack_bars(bounce=True)
    sigs = detect_signals(df)
    trades = simulate(df, sigs)
    assert trades
    assert isinstance(trades[0], TradeResult)
    out_dir = Path("/tmp/nq_5m_base_stack_test") if tmp_path is None else Path(tmp_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = write_html_report(out_dir / "r.html", df, trades, "NQ=F", "demo")
    text = path.read_text(encoding="utf-8")
    assert "打底" in text
    assert "多排" in text
    if trades:
        assert "<img src='img/" in text
        img_dir = path.parent / "img"
        pngs = list(img_dir.glob("t01_*.png"))
        assert pngs, "expected a static trade PNG"
        assert "五分 K 參考" in text
        import struct

        with pngs[0].open("rb") as fh:
            fh.read(16)
            w, h = struct.unpack(">II", fh.read(8))
        assert h > w * 0.6, "trade card should include a 5m reference pane under the zoom"


def test_ref_window_wider_than_zoom() -> None:
    df = _make_base_stack_bars(bounce=True)
    trades = simulate(df, detect_signals(df))
    assert trades
    z0, z1 = _zoom_window(df, trades[0])
    r0, r1 = _ref_window(df, trades[0])
    assert r0 <= z0 and r1 >= z1
    assert (r1 - r0) > (z1 - z0)


def test_deep_dump_risk_uses_half_drop() -> None:
    """07-22 style: dump ~240, first fan is >80pts above the low — old 80-cap missed it."""
    df = _make_base_stack_bars(bounce=True, deep=True)
    blocked = detect_signals(df, max_risk=80.0, max_risk_frac=0.0)
    assert not blocked, "hard 80-pt cap should still reject the deep bounce"
    sigs = detect_signals(df)
    assert sigs, "half-drop risk cap should take the 07-22-style fan"
    risk = sigs[0].entry_price - sigs[0].stop_price
    assert risk > 80.0
    assert risk <= max(80.0, 0.50 * sigs[0].drop_pts)


def test_no_signal_on_knot_stack() -> None:
    df = _make_base_stack_bars(bounce=True, knot=True)
    sigs = detect_signals(df)
    assert not sigs, "sticky MA kiss after the low is not 多排"


def test_no_signal_on_slow_dump() -> None:
    df = _make_base_stack_bars(bounce=True, slow=True)
    sigs = detect_signals(df)
    assert not sigs, "a grind-down pause should not count as the screenshot U"


def test_loose_detects_knot() -> None:
    df = _make_base_stack_bars(bounce=True, knot=True)
    assert not detect_signals(df)
    sigs = detect_signals(df, **LOOSE_DETECT)
    assert sigs, "loose mode should still take a 5/10/20 flip even if the ribbon is a knot"


def test_html_extra_blurb(tmp_path: Path | None = None) -> None:
    df = _make_base_stack_bars(bounce=True)
    trades = simulate(df, detect_signals(df))
    out = Path("/tmp/nq_5m_base_stack_extra.html") if tmp_path is None else Path(tmp_path) / "e.html"
    path = write_html_report(
        out,
        df,
        trades,
        "NQ=F",
        "demo",
        extra_trades=trades,
        extra_title="嚴格（截圖那種 U）",
        extra_blurb="急跌集中、短打底、均線散開上攻。",
        blurb="放寬版說明",
    )
    text = path.read_text(encoding="utf-8")
    assert "放寬版說明" in text
    assert "嚴格（截圖那種 U）" in text
    assert "急跌集中、短打底、均線散開上攻。" in text


def main() -> int:
    test_quality_from_setup()
    test_sma()
    test_summarize_trades()
    test_detect_base_then_stack()
    test_deep_dump_risk_uses_half_drop()
    test_no_signal_on_continued_dump()
    test_no_signal_on_knot_stack()
    test_no_signal_on_slow_dump()
    test_loose_detects_knot()
    test_simulate_and_html()
    test_ref_window_wider_than_zoom()
    test_html_extra_blurb()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

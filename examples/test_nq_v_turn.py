#!/usr/bin/env python3
"""Synthetic tests for NQ 5m V轉 (no Yahoo)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nq_v_turn import (  # noqa: E402
    ET,
    TradeResult,
    detect_signals,
    quality_from_v,
    simulate,
    summarize_trades,
    write_html_report,
)


def test_quality_from_v() -> None:
    assert quality_from_v(3.0, 1.2, 1.5)[1] == "A"
    assert quality_from_v(2.0, 0.5, 1.0)[1] == "C"
    assert quality_from_v(2.6, 0.5, 1.0)[1] == "B"


def test_summarize_trades() -> None:
    class T:
        def __init__(self, pnl: float, quality: str = "A"):
            self.pnl_points = pnl
            self.quality = quality

    stats = summarize_trades([T(20.0, "A"), T(-8.0, "B")])
    assert stats["count"] == 2
    assert stats["wins"] == 1
    assert abs(stats["total_points"] - 12.0) < 1e-9


def _make_v_bars(
    *,
    bounce: bool = True,
    u_base: bool = False,
    fast_recover: bool = False,
    n: int = 220,
) -> pd.DataFrame:
    """~24-bar 150pt dump, sharp pivot, then a matching-length climb back to the neckline."""
    close = np.zeros(n, dtype=float)
    close[0] = 29540.0
    for i in range(1, 90):
        close[i] = close[i - 1] + (1.2 if i % 3 else -0.4)

    dump_start = 100
    dump_end = 123
    high_level = float(close[dump_start - 1])
    dump_len = dump_end - dump_start + 1
    for i in range(dump_start, dump_end + 1):
        frac = (i - dump_start + 1) / dump_len
        close[i] = high_level - 150.0 * frac

    if u_base:
        close[dump_end + 1 : dump_end + 9] = close[dump_end] + np.array(
            [2, -1, 3, -2, 1, 2, -1, 2], dtype=float
        )
        tail = dump_end + 9
    else:
        tail = dump_end

    pivot = float(close[dump_end])
    if bounce:
        rec_len = 5 if fast_recover else dump_len
        for k in range(1, rec_len + 1):
            i = tail + k
            if i >= n:
                break
            close[i] = pivot + (high_level - pivot) * (k / rec_len)
        last = tail + rec_len
        if last < n:
            close[last] = high_level + 10.0
        for i in range(last + 1, n):
            close[i] = close[i - 1] + 0.4
    else:
        for i in range(tail + 1, n):
            close[i] = close[i - 1] - 6.0

    high = close + 2.0
    low = close - 2.0
    open_ = np.r_[close[0], close[:-1]]

    for i in range(dump_start, dump_end + 1):
        open_[i] = close[i] + 6.0
        high[i] = open_[i] + 0.5
        low[i] = close[i] - 2.5
    if bounce and not u_base:
        open_[dump_end] = close[dump_end] + 4.0
        low[dump_end] = close[dump_end] - 1.5
        high[dump_end] = close[dump_end] + 8.0
        rec_len = 5 if fast_recover else dump_len
        for k in range(1, rec_len + 1):
            i = tail + k
            if i >= n:
                break
            open_[i] = close[i] - 4.0
            high[i] = close[i] + 3.0
            low[i] = min(close[i] - 2.0, open_[i] - 1.0)

    vol = np.full(n, 90.0)
    vol[dump_start : dump_end + 6] = 240.0

    idx = pd.date_range("2026-08-27 00:00", periods=n, freq="5min", tz=ET)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=idx,
    )


def test_detect_v_bounce() -> None:
    df = _make_v_bars(bounce=True)
    sigs = detect_signals(df, skip_rth_open=False)
    assert sigs, "expected a V-turn after the synthetic dump"
    sig = sigs[0]
    assert sig.entry_idx > sig.dump_idx
    assert sig.dump_low < sig.dump_high
    assert sig.drop_pts >= 120
    assert sig.recover_frac >= 0.98
    assert 0.65 <= sig.time_ratio <= 1.40
    assert sig.entry_price > sig.stop_price
    assert sig.entry_price > sig.ma5
    assert sig.target_price > sig.entry_price


def test_no_signal_on_continued_dump() -> None:
    df = _make_v_bars(bounce=False)
    sigs = detect_signals(df, skip_rth_open=False)
    assert not sigs, "waterfall should not print a V"


def test_no_signal_on_u_base() -> None:
    df = _make_v_bars(bounce=True, u_base=True)
    sigs = detect_signals(df, skip_rth_open=False)
    assert not sigs, "rounded U-base should not count as a V"


def test_no_signal_on_asymmetric_spike() -> None:
    df = _make_v_bars(bounce=True, fast_recover=True)
    sigs = detect_signals(df, skip_rth_open=False)
    assert not sigs, "5-bar spike back is not a symmetric neckline V"


def test_simulate_and_html(tmp_path: Path | None = None) -> None:
    df = _make_v_bars(bounce=True)
    sigs = detect_signals(df, skip_rth_open=False)
    trades = simulate(df, sigs)
    assert trades
    assert isinstance(trades[0], TradeResult)
    out = Path("/tmp/nq_v_turn_test.html") if tmp_path is None else Path(tmp_path) / "r.html"
    path = write_html_report(out, df, trades, "NQ=F", "demo")
    text = path.read_text(encoding="utf-8")
    assert "V轉" in text
    assert "頸線" in text
    if trades:
        assert "<img src='img/" in text
        img_dir = path.parent / "img"
        assert any(img_dir.glob("t01_*.png")), "expected a static trade PNG"


def main() -> int:
    test_quality_from_v()
    test_summarize_trades()
    test_detect_v_bounce()
    test_no_signal_on_continued_dump()
    test_no_signal_on_u_base()
    test_no_signal_on_asymmetric_spike()
    test_simulate_and_html()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

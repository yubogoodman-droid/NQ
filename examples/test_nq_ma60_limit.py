#!/usr/bin/env python3
"""Synthetic tests for NQ 1m 破底後掛單 MA60 五分鐘 (no Yahoo)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nq_ma60_limit import (  # noqa: E402
    ET,
    PendingOrder,
    detect_signals,
    is_tangled,
    simulate,
)
from nq_ma_reclaim import (  # noqa: E402
    TradeResult,
    sma,
    write_html_report,
)


def _to_df(close, high, low) -> pd.DataFrame:
    n = len(close)
    idx = pd.date_range("2026-08-17 11:00", periods=n, freq="1min", tz=ET)
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


def _base_dump(n: int = 320) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    """Range to pin the 2h low, then fake-break. Post-break bars stay near the dump until painted."""
    close = np.zeros(n, dtype=float)
    close[0] = 20000.0
    for i in range(1, 100):
        close[i] = close[i - 1] + 0.55
    base = close[99]
    for i in range(100, 220):
        close[i] = base + (1.0 if i % 2 == 0 else -0.4)
    break_i = 220
    close[break_i] = base - 18.0
    for i in range(break_i + 1, n):
        close[i] = base - 16.0
    high = close + 0.8
    low = close - 0.8
    for i in range(100, 220):
        low[i] = min(close[i] - 0.3, base - 1.5)
        high[i] = close[i] + 0.3
    low[break_i] = close[break_i] - 0.5
    high[break_i] = close[break_i] + 0.5
    return close, high, low, base, break_i


def _paint_bounce(
    close,
    high,
    low,
    base: float,
    break_i: int,
    *,
    delay: int = 1,
    hold: int = 12,
    under: int = 16,
    dump: float = 36.0,
    tangle: bool = False,
) -> None:
    """Stay under MA60, then reclaim. `tangle=True` keeps short MAs glued (圖一)."""
    if tangle:
        under, dump = 6, 18.0
    for i in range(delay, delay + under):
        j = break_i + i
        if j >= len(close):
            return
        close[j] = base - dump + i * 0.35
        high[j] = close[j] + 0.8
        low[j] = close[j] - 0.8
    start = break_i + delay + under
    if tangle:
        steps = (base - 6.0, base - 4.0, base - 2.0, base + 2.0, base + 8.0, base + 14.0)
    else:
        # hold below MA60 so MA5 lifts off MA20/30, then punch through
        steps = (base - 18.0, base - 16.0, base - 14.0, base - 12.0, base - 10.0, base + 6.0, base + 16.0)
    for i, px in enumerate(steps):
        j = start + i
        if j >= len(close):
            return
        close[j] = px
        high[j] = px + 1.2
        low[j] = px - 1.2
    rest = start + len(steps)
    for i in range(rest, len(close)):
        close[i] = close[i - 1] + (1.6 if i - rest < hold else 0.8)
        high[i] = close[i] + 1.0
        low[i] = close[i] - 0.8


def _first_setup(close, break_i: int, window: int = 30) -> int | None:
    ma5 = sma(close, 5)
    ma20 = sma(close, 20)
    ma60 = sma(close, 60)
    last = min(break_i + window, len(close) - 1)
    for j in range(break_i + 1, last + 1):
        if np.isnan(ma5[j]) or np.isnan(ma20[j]) or np.isnan(ma60[j]) or np.isnan(ma60[j - 1]):
            continue
        if float(ma5[j]) <= float(ma20[j]):
            continue
        if float(close[j]) > float(ma60[j]) and float(close[j - 1]) <= float(ma60[j - 1]):
            return j
    return None


def _make_fill_bars(n: int = 320) -> pd.DataFrame:
    close, high, low, base, break_i = _base_dump(n)
    _paint_bounce(close, high, low, base, break_i)
    setup = _first_setup(close, break_i)
    assert setup is not None, "synthetic bounce must break MA60 with MA5>MA20"
    ma60 = sma(close, 60)
    fill_i = setup + 7
    limit = float(ma60[setup])
    for i in range(setup + 1, fill_i):
        close[i] = limit + 12.0
        high[i] = close[i] + 1.0
        low[i] = limit + 8.0
    close[fill_i] = limit + 3.0
    high[fill_i] = close[fill_i] + 1.0
    low[fill_i] = limit - 0.8
    for i in range(fill_i + 1, n):
        close[i] = close[i - 1] + 1.4
        high[i] = close[i] + 1.0
        low[i] = close[i] - 0.6
    return _to_df(close, high, low)


def _make_expire_bars(n: int = 320) -> pd.DataFrame:
    close, high, low, base, break_i = _base_dump(n)
    _paint_bounce(close, high, low, base, break_i)
    setup = _first_setup(close, break_i)
    assert setup is not None
    ma60 = sma(close, 60)
    limit = float(ma60[setup])
    for i in range(setup + 1, n):
        close[i] = max(float(close[i]), limit + 22.0)
        low[i] = limit + 18.0
        high[i] = close[i] + 1.2
    return _to_df(close, high, low)


def _make_late_setup_bars(n: int = 340) -> pd.DataFrame:
    """MA60 break after 30 minutes — should not hang an order."""
    close, high, low, base, break_i = _base_dump(n)
    for i in range(break_i + 1, break_i + 32):
        close[i] = base - 14.0
        high[i] = close[i] + 0.6
        low[i] = close[i] - 0.6
    _paint_bounce(close, high, low, base, break_i, delay=32, hold=10)
    setup = _first_setup(close, break_i, window=80)
    if setup is not None:
        ma60 = sma(close, 60)
        fill_i = setup + 2
        limit = float(ma60[setup])
        close[fill_i] = limit + 2.0
        high[fill_i] = close[fill_i] + 1.0
        low[fill_i] = limit - 1.0
    return _to_df(close, high, low)


def test_fill_within_five_minutes() -> None:
    df = _make_fill_bars()
    funnel: dict[str, int] = {}
    sigs = detect_signals(df, funnel=funnel)
    assert sigs, f"expected a fill, funnel={funnel}"
    sig = sigs[0]
    assert sig.setup_idx > sig.break_idx
    assert sig.entry_idx > sig.setup_idx
    assert sig.entry_idx - sig.setup_idx > 5
    assert sig.entry_idx - sig.setup_idx <= 10
    assert sig.limit_price > 0
    assert abs(sig.entry_price - sig.limit_price) < 2.0 or sig.entry_price <= sig.limit_price
    assert abs(sig.stop_price - sig.break_low) < 1e-9
    assert sig.entry_price > sig.stop_price
    trades = simulate(df, sigs, preopen_flat=False, stop_on_m5_close=False)
    assert trades
    assert isinstance(trades[0], TradeResult)
    assert all(t.exit_reason != "ma60_stop" for t in trades)


def _make_early_only_bars(n: int = 320) -> pd.DataFrame:
    """回踩只發生在突破後 2 根 — 五根內，應不算。"""
    close, high, low, base, break_i = _base_dump(n)
    _paint_bounce(close, high, low, base, break_i)
    setup = _first_setup(close, break_i)
    assert setup is not None
    ma60 = sma(close, 60)
    limit = float(ma60[setup])
    early = setup + 2
    for i in range(setup + 1, early):
        close[i] = limit + 10.0
        high[i] = close[i] + 1.0
        low[i] = limit + 8.0
    close[early] = limit + 2.0
    high[early] = close[early] + 1.0
    low[early] = limit - 1.0
    for i in range(early + 1, n):
        close[i] = limit + 18.0
        high[i] = close[i] + 1.2
        low[i] = limit + 14.0
    return _to_df(close, high, low)


def test_early_retest_ignored() -> None:
    df = _make_early_only_bars()
    funnel: dict[str, int] = {}
    sigs = detect_signals(df, funnel=funnel)
    assert not sigs, f"retest inside 5 bars must not fill, funnel={funnel}"
    assert funnel.get("skip_early", 0) >= 1
    assert funnel.get("taken", 0) == 0


def test_expire_if_no_retest() -> None:
    df = _make_expire_bars()
    funnel: dict[str, int] = {}
    pending: list[PendingOrder] = []
    sigs = detect_signals(df, funnel=funnel, pending=pending)
    assert not sigs, f"no pullback after the 5-bar wait should not fill, funnel={funnel}"
    assert funnel.get("setup", 0) >= 1
    assert funnel.get("expired", 0) >= 1
    assert not pending


def test_late_ma60_break_ignored() -> None:
    df = _make_late_setup_bars()
    funnel: dict[str, int] = {}
    sigs = detect_signals(df, funnel=funnel)
    assert not sigs, f"setup after 30m must be ignored, funnel={funnel} sigs={sigs}"


def _make_tangle_bars(n: int = 320) -> pd.DataFrame:
    """MA60 break while MA5/10/20/30 are still glued — 圖一。"""
    close, high, low, base, break_i = _base_dump(n)
    _paint_bounce(close, high, low, base, break_i, tangle=True)
    setup = _first_setup(close, break_i)
    assert setup is not None, "need an MA60 cross to test the tangle skip"
    ma60 = sma(close, 60)
    fill_i = setup + 2
    limit = float(ma60[setup])
    close[fill_i] = limit + 2.0
    high[fill_i] = close[fill_i] + 1.0
    low[fill_i] = limit - 1.0
    return _to_df(close, high, low)


def test_skip_tangled_mas() -> None:
    assert is_tangled(100.0, 101.0, 99.5, 100.5, 12.0)
    assert not is_tangled(100.0, 108.0, 92.0, 95.0, 12.0)
    df = _make_tangle_bars()
    funnel: dict[str, int] = {}
    sigs = detect_signals(df, funnel=funnel)
    assert not sigs, f"tangled short MAs must not fill, funnel={funnel}"
    assert funnel.get("skip_tangle", 0) >= 1


def test_pending_at_end_of_data() -> None:
    df = _make_fill_bars()
    setup_guess = None
    close = df["Close"].to_numpy(float)
    # cut during the 5-bar wait so the order is still pending
    ma5 = sma(close, 5)
    ma20 = sma(close, 20)
    ma60 = sma(close, 60)
    for j in range(221, len(df)):
        if np.isnan(ma60[j]) or np.isnan(ma60[j - 1]):
            continue
        if float(ma5[j]) > float(ma20[j]) and float(close[j]) > float(ma60[j]) and float(close[j - 1]) <= float(ma60[j - 1]):
            setup_guess = j
            break
    assert setup_guess is not None
    cut = df.iloc[: setup_guess + 2].copy()
    pending: list[PendingOrder] = []
    funnel: dict[str, int] = {}
    sigs = detect_signals(cut, funnel=funnel, pending=pending)
    assert not sigs
    assert pending, f"expected a live 掛單, funnel={funnel}"
    assert pending[0].setup_idx == setup_guess
    assert pending[0].expire_idx == setup_guess + 10


def test_write_html_report(tmp_path: Path | None = None) -> None:
    df = _make_fill_bars()
    sigs = detect_signals(df)
    trades = simulate(df, sigs, preopen_flat=False, stop_on_m5_close=False)
    out = Path("/tmp/nq_ma60_limit_test.html") if tmp_path is None else Path(tmp_path) / "r.html"
    path = write_html_report(
        out,
        df,
        trades,
        "NQ=F",
        "demo",
        title="破底後掛單 MA60",
        rules="破 2h 低後 30 分鐘內掛單 MA60 五分鐘",
    )
    text = path.read_text(encoding="utf-8")
    assert "掛單 MA60" in text
    if trades:
        assert "<img src='img/" in text
        img_dir = path.parent / "img"
        assert any(img_dir.glob("t01_*.png")), "expected a static trade PNG"


def main() -> int:
    test_fill_within_five_minutes()
    test_early_retest_ignored()
    test_expire_if_no_retest()
    test_late_ma60_break_ignored()
    test_skip_tangled_mas()
    test_pending_at_end_of_data()
    test_write_html_report()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

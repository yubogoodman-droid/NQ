#!/usr/bin/env python3
"""Synthetic tests for 1h MA25 undercut + reclaim (no Binance)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from binance_1h_ma25_reclaim import (  # noqa: E402
    MA_COLORS,
    Hit,
    TradeResult,
    atr,
    classify_shape,
    detect_signals,
    flush_metrics,
    one_at_a_time_path,
    quality_of,
    resample_4h,
    simulate,
    sma,
    summarize_trades,
    write_html_report,
    write_seq_html,
)


def test_resample_4h() -> None:
    start = pd.Timestamp("2026-08-27 08:00", tz="Asia/Taipei")  # 00:00 UTC
    times = [start + pd.Timedelta(hours=i) for i in range(8)]
    df = pd.DataFrame(
        {
            "Open": [100, 101, 102, 99, 98, 97, 96, 100],
            "High": [101, 103, 104, 100, 99, 98, 97, 105],
            "Low": [99, 100, 98, 95, 96, 94, 93, 99],
            "Close": [101, 102, 99, 98, 97, 96, 100, 104],
            "Volume": [1, 2, 3, 4, 5, 6, 7, 8],
        },
        index=times,
    )
    out = resample_4h(df)
    assert len(out) == 2
    first = out.iloc[0]
    assert first["Open"] == 100
    assert first["High"] == 104
    assert first["Low"] == 95
    assert first["Close"] == 98
    assert first["Volume"] == 10
    assert out.index[0] == start
    assert out.index[1] == start + pd.Timedelta(hours=4)


def test_sma() -> None:
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = sma(arr, 3)
    assert np.isnan(out[1])
    assert abs(out[2] - 2.0) < 1e-9
    assert abs(out[4] - 4.0) < 1e-9


def test_quality_of() -> None:
    assert quality_of(0.03, 1.5, "W", True)[1] == "A"
    assert quality_of(0.03, 1.0, "V", False)[1] == "C"
    assert quality_of(0.03, 1.5, "V", False)[1] == "B"


def test_classify_v_and_w() -> None:
    n = 20
    low = np.full(n, 10.0)
    high = np.full(n, 10.4)
    low[8] = 9.4
    assert classify_shape(low, high, 4, 16) == "V"

    low2 = np.full(n, 10.0)
    high2 = np.full(n, 10.5)
    low2[6] = 9.50
    high2[10] = 10.80
    low2[14] = 9.52
    assert classify_shape(low2, high2, 4, 17) == "W"


def _base_index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2026-08-01 00:00", periods=n, freq="1h", tz="Asia/Taipei")


def _make_reclaim_bars(n: int = 80, *, depth: float = 0.055, below: int = 14, sharp: bool = True) -> pd.DataFrame:
    """Ride above MA25, sit under it, then a 4-bar washout like the screenshots."""
    close = np.zeros(n, dtype=float)
    close[0] = 100.0
    for i in range(1, 40):
        close[i] = close[i - 1] + 0.12
    dump = 40
    base = close[dump - 1]
    floor = base * (1.0 - depth)
    hang = base * 0.978  # 明確掛在 MA25 下，但還沒急殺
    for i in range(dump, dump + below):
        close[i] = hang
    if sharp:
        flush_at = dump + max(below - 6, 4)
        close[flush_at] = hang * 0.997
        close[flush_at + 1] = hang * 0.985
        close[flush_at + 2] = floor * 1.004
        close[flush_at + 3] = floor * 1.012
        close[dump + below - 1] = hang
    else:
        for i in range(dump, dump + below):
            t = (i - dump) / max(below - 1, 1)
            close[i] = base + (floor - base) * t
        close[dump + below - 1] = base * 0.985
    close[dump + below] = base + 0.80
    for i in range(dump + below + 1, n):
        close[i] = close[i - 1] + 0.25
    high = close + 0.35
    low = close - 0.35
    if sharp:
        flush_at = dump + max(below - 6, 4)
        low[flush_at + 2] = floor
        high[flush_at + 2] = close[flush_at + 2] + 0.10
    return pd.DataFrame(
        {
            "Open": np.r_[close[0], close[:-1]],
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": np.r_[np.full(dump + below, 80.0), [220.0], np.full(n - dump - below - 1, 90.0)],
        },
        index=_base_index(n),
    )


def test_detect_reclaim() -> None:
    df = _make_reclaim_bars()
    funnel: dict = {}
    sigs = detect_signals(df, funnel=funnel)
    assert sigs, f"expected a reclaim, funnel={funnel}"
    sig = sigs[0]
    assert sig.entry_idx > sig.break_idx
    assert sig.entry_price > sig.stop_price
    assert sig.bars_below >= 4
    assert sig.depth_pct >= 0.018
    assert sig.quality in {"A", "B", "C"}
    assert df["Close"].iloc[sig.entry_idx] > sig.ma25
    assert abs(sig.stop_price - sig.bottom) < 1e-12


def test_flush_metrics_avgo_like() -> None:
    n = 20
    high = np.full(n, 358.0)
    low = np.full(n, 356.5)
    close = np.full(n, 357.2)
    high[16:20] = [359.70, 357.00, 354.00, 353.00]
    low[16:20] = [356.00, 354.00, 350.39, 351.20]
    close[16:20] = [357.10, 352.80, 351.70, 352.10]
    a = atr(high, low, close)
    imp, flush_atr, under = flush_metrics(high, low, 18, 350.39, a)
    assert abs(imp - (359.70 - 350.39) / 359.70) < 1e-9
    assert under >= 0.014
    assert flush_atr >= 2.8


def test_shallow_rejected() -> None:
    df = _make_reclaim_bars(depth=0.006, below=8)
    sigs = detect_signals(df, min_depth_pct=0.028)
    assert not sigs, "0.6% poke should not count as 破底"


def test_slow_grind_rejected() -> None:
    df = _make_reclaim_bars(depth=0.055, below=16, sharp=False)
    funnel: dict = {}
    sigs = detect_signals(
        df,
        funnel=funnel,
        min_bars_below=10,
        min_depth_pct=0.028,
        min_impulse_pct=0.023,
        min_undercut_pct=0.014,
        min_flush_atr=2.8,
    )
    assert not sigs, f"slow grind should fail flush, funnel={funnel}"
    assert funnel.get("weak_flush", 0) >= 1 or funnel.get("shallow", 0) >= 1 or funnel.get("too_short", 0) >= 1


def test_one_bar_pop_keeps_episode() -> None:
    df = _make_reclaim_bars(depth=0.055, below=16)
    close = df["Close"].to_numpy(float)
    # W 中間兩根假站上，然後再破底
    df.iloc[44, df.columns.get_loc("Close")] = close[39] + 0.4
    df.iloc[44, df.columns.get_loc("High")] = close[39] + 0.6
    df.iloc[45, df.columns.get_loc("Close")] = close[39] + 0.25
    df.iloc[45, df.columns.get_loc("High")] = close[39] + 0.45
    sigs = detect_signals(df)
    assert sigs, "a 1–2 bar bounce above MA25 should not split the dip"
    assert sigs[0].bars_below >= 8


def test_still_below_no_signal() -> None:
    df = _make_reclaim_bars()
    # chop the reclaim bar off
    cut = df.iloc[:50].copy()
    # force the tail to stay under a rising MA
    cut.loc[cut.index[-6]:, "Close"] = 90.0
    cut.loc[cut.index[-6]:, "Low"] = 89.5
    cut.loc[cut.index[-6]:, "High"] = 90.4
    sigs = detect_signals(cut)
    assert not sigs


def _flatten_after_entry(df: pd.DataFrame, sig, bars: int = 8) -> None:
    """Hold near entry so later bars don't hit 2R or MA25 exit by accident."""
    px = float(sig.entry_price)
    end = min(sig.entry_idx + bars, len(df) - 1)
    for i in range(sig.entry_idx + 1, end + 1):
        df.iloc[i, df.columns.get_loc("Open")] = px
        df.iloc[i, df.columns.get_loc("High")] = px + 0.15
        df.iloc[i, df.columns.get_loc("Low")] = px - 0.15
        df.iloc[i, df.columns.get_loc("Close")] = px + 0.05


def test_wick_below_bottom_bar_does_not_stop() -> None:
    df = _make_reclaim_bars()
    sigs = detect_signals(df)
    assert sigs
    sig = sigs[0]
    _flatten_after_entry(df, sig)
    i = sig.entry_idx + 1
    df.iloc[i, df.columns.get_loc("Low")] = sig.bottom - 2.0
    df.iloc[i, df.columns.get_loc("Close")] = sig.entry_price
    df.iloc[i, df.columns.get_loc("Open")] = sig.entry_price
    df.iloc[i, df.columns.get_loc("High")] = sig.entry_price + 0.15
    trades = simulate(df, [sig], max_hold=6)
    assert trades
    assert trades[0].exit_reason != "stop", "影線掃破底K低點不該停損"


def test_close_below_bottom_bar_stops() -> None:
    df = _make_reclaim_bars()
    sigs = detect_signals(df)
    assert sigs
    sig = sigs[0]
    _flatten_after_entry(df, sig)
    i = sig.entry_idx + 2
    fill = sig.bottom - 0.40
    df.iloc[i, df.columns.get_loc("Open")] = sig.entry_price
    df.iloc[i, df.columns.get_loc("High")] = sig.entry_price
    df.iloc[i, df.columns.get_loc("Low")] = fill - 0.10
    df.iloc[i, df.columns.get_loc("Close")] = fill
    trades = simulate(df, [sig], max_hold=6)
    assert trades
    assert trades[0].exit_reason == "stop"
    assert trades[0].exit_idx == i
    assert abs(trades[0].exit_price - fill) < 1e-9


def test_simulate_target() -> None:
    df = _make_reclaim_bars()
    sigs = detect_signals(df)
    trades = simulate(df, sigs, max_hold=40)
    assert trades
    assert isinstance(trades[0], TradeResult)
    assert trades[0].exit_idx >= trades[0].entry_idx
    assert trades[0].exit_reason in {"target", "stop", "lost_ma25", "timeout"}


def test_summarize() -> None:
    class T:
        def __init__(self, pnl: float, quality: str = "A"):
            self.pnl_pct = pnl
            self.quality = quality

    stats = summarize_trades([T(2.0, "A"), T(-1.0, "B")])  # type: ignore[list-item]
    assert stats["count"] == 2
    assert stats["wins"] == 1
    assert abs(stats["total_pct"] - 1.0) < 1e-9


def test_select_card_hits() -> None:
    from binance_1h_ma25_reclaim import Hit, select_card_hits  # noqa: WPS433

    df = _make_reclaim_bars()
    sigs = detect_signals(df)
    trades = simulate(df, sigs, max_hold=40)
    assert trades
    hits = [Hit("FOOUSDT", df, trades[0]), Hit("AVGOUSDT", df, trades[0])]
    picked = select_card_hits(hits, recent_hours=0, keep_symbols=("AVGOUSDT",))
    assert [h.symbol for h in picked] == ["AVGOUSDT"]


def test_one_at_a_time_skips_overlap() -> None:
    df = _make_reclaim_bars()
    sigs = detect_signals(df)
    trades = simulate(df, sigs, max_hold=40)
    assert trades
    t = trades[0]
    later = TradeResult(
        signal=t.signal,
        entry_idx=min(t.entry_idx + 1, len(df) - 2),
        exit_idx=min(t.exit_idx + 1, len(df) - 1),
        entry_price=t.entry_price,
        exit_price=t.exit_price,
        stop_price=t.stop_price,
        target_price=t.target_price,
        pnl_pct=10.0,
        exit_reason="target",
        quality=t.quality,
    )
    hits = [Hit("AAAUSDT", df, t), Hit("BBBUSDT", df, later)]
    path = one_at_a_time_path(hits, start=100.0, lev=3.0)
    assert len(path) == 1
    assert path[0]["symbol"] == "AAAUSDT"
    assert abs(path[0]["after"] - 100.0 * (1.0 + 3.0 * t.pnl_pct / 100.0)) < 1e-6


def test_write_html(tmp_path: Path | None = None) -> None:
    df = _make_reclaim_bars()
    sigs = detect_signals(df)
    trades = simulate(df, sigs, max_hold=40)
    from binance_1h_ma25_reclaim import Hit  # noqa: WPS433

    out = Path("/tmp/ma25_reclaim_test.html") if tmp_path is None else Path(tmp_path) / "r.html"
    hits = [Hit("AVGOUSDT", df, t) for t in trades]
    path = write_html_report(out, hits, days=7, scanned=1)
    text = path.read_text(encoding="utf-8")
    assert "MA25" in text
    assert "下破底" in text
    assert "1h + 4h" in text
    assert "4h K 對照" in text
    assert "收盤跌破破底" in text
    assert "黃7" in text
    assert "酒紅200" in text
    assert set(MA_COLORS) == {7, 14, 25, 99, 120, 200}
    seq_hits = [Hit("AVGOUSDT", df, t) for t in trades]
    seq_path = one_at_a_time_path(seq_hits, start=100.0, lev=3.0)
    seq = Path("/tmp/ma25_seq_test.html") if tmp_path is None else Path(tmp_path) / "seq.html"
    write_seq_html(seq, seq_path, days=7, start=100.0, lev=3.0, scanned=1, pool=len(seq_hits))
    seq_text = seq.read_text(encoding="utf-8")
    assert "一次一單" in seq_text
    assert "<svg" in seq_text
    if trades:
        assert "<img src='img/" in text
        assert any((path.parent / "img").glob("t01_*.png"))


def main() -> int:
    test_resample_4h()
    test_sma()
    test_quality_of()
    test_classify_v_and_w()
    test_detect_reclaim()
    test_flush_metrics_avgo_like()
    test_shallow_rejected()
    test_slow_grind_rejected()
    test_still_below_no_signal()
    test_wick_below_bottom_bar_does_not_stop()
    test_close_below_bottom_bar_stops()
    test_simulate_target()
    test_summarize()
    test_select_card_hits()
    test_one_at_a_time_skips_overlap()
    test_write_html()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

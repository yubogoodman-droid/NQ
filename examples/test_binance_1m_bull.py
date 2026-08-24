#!/usr/bin/env python3
"""Synthetic tests for 1m MA7>14>25>99>120 above MA200 stack (no Binance)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nq.ma1m_bull import (  # noqa: E402
    add_mas,
    detect_combo,
    forward_moves,
    sma,
    stack_ok,
    summarize_rows,
    SignalRow,
)


def test_sma() -> None:
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = sma(arr, 3)
    assert np.isnan(out[1])
    assert abs(out[2] - 2.0) < 1e-9
    assert abs(out[4] - 4.0) < 1e-9


def _make_stack_bars(n: int = 280) -> dict:
    """Slow grind then a lift so short MAs stack above MA200."""
    close = np.zeros(n, dtype=float)
    close[0] = 100.0
    for i in range(1, 200):
        close[i] = close[i - 1] + (0.02 if i % 3 else -0.01)
    for i in range(200, n):
        close[i] = close[i - 1] + 0.35
    rng = np.random.default_rng(0)
    high = close + 0.08
    low = close - 0.08
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    vol = rng.uniform(100, 200, n)
    t0 = 1_700_000_000_000
    return {
        "t": np.arange(n, dtype=np.int64) * 60_000 + t0,
        "o": open_,
        "h": high,
        "l": low,
        "c": close,
        "v": vol,
    }


def test_detects_first_stack_above_ma200() -> None:
    d = add_mas(_make_stack_bars())
    sigs = detect_combo(d)
    assert len(sigs) >= 1
    first = sigs[0]
    assert first.idx >= 199
    assert stack_ok(d, first.idx)
    assert not stack_ok(d, first.idx - 1)
    assert d["c"][first.idx] > d["m7"][first.idx] > d["m14"][first.idx]
    assert d["m14"][first.idx] > d["m25"][first.idx] > d["m99"][first.idx] > d["m120"][first.idx]
    assert d["c"][first.idx] > d["m200"][first.idx]


def test_no_repeat_while_stack_holds() -> None:
    d = add_mas(_make_stack_bars())
    sigs = detect_combo(d)
    # After the first print, the grind keeps the stack; should not spam every bar.
    assert len(sigs) == 1


def test_rearms_after_break() -> None:
    raw = _make_stack_bars(360)
    # After the first stack, dump below MA200 then lift again.
    for i in range(250, 280):
        raw["c"][i] = raw["c"][249] - (i - 249) * 0.8
        raw["h"][i] = raw["c"][i] + 0.08
        raw["l"][i] = raw["c"][i] - 0.08
        raw["o"][i] = raw["c"][i - 1]
    for i in range(280, 360):
        raw["c"][i] = raw["c"][i - 1] + 0.45
        raw["h"][i] = raw["c"][i] + 0.08
        raw["l"][i] = raw["c"][i] - 0.08
        raw["o"][i] = raw["c"][i - 1]
    d = add_mas(raw)
    sigs = detect_combo(d)
    assert len(sigs) >= 2
    assert sigs[1].idx > sigs[0].idx + 10


def test_min_gap() -> None:
    raw = _make_stack_bars(360)
    for i in range(250, 260):
        raw["c"][i] = raw["c"][249] - (i - 249) * 0.9
        raw["h"][i] = raw["c"][i] + 0.08
        raw["l"][i] = raw["c"][i] - 0.08
        raw["o"][i] = raw["c"][i - 1]
    for i in range(260, 360):
        raw["c"][i] = raw["c"][i - 1] + 0.5
        raw["h"][i] = raw["c"][i] + 0.08
        raw["l"][i] = raw["c"][i] - 0.08
        raw["o"][i] = raw["c"][i - 1]
    d = add_mas(raw)
    all_sigs = detect_combo(d, min_gap_bars=0)
    gapped = detect_combo(d, min_gap_bars=80)
    assert len(all_sigs) >= 2
    assert len(gapped) == 1


def test_forward_and_summarize() -> None:
    d = add_mas(_make_stack_bars())
    sigs = detect_combo(d)
    entry, moves = forward_moves(d, sigs[0])
    assert entry > 0
    assert moves[5].ret_pct is not None
    assert moves[5].ret_pct > 0
    row = SignalRow(symbol="TESTUSDT", sig=sigs[0], time_ms=int(d["t"][sigs[0].idx]), entry=entry, moves=moves)
    stats = summarize_rows([row], 5)
    assert stats["n"] == 1
    assert stats["wr"] == 100.0


def test_cross_only_keeps_ma200_reclaim() -> None:
    raw = _make_stack_bars(300)
    raw["c"][:220] = 100.0
    raw["o"][:220] = 100.0
    raw["h"][:220] = 100.08
    raw["l"][:220] = 99.92
    for i in range(220, 300):
        raw["c"][i] = raw["c"][i - 1] + 0.40
        raw["h"][i] = raw["c"][i] + 0.08
        raw["l"][i] = raw["c"][i] - 0.08
        raw["o"][i] = raw["c"][i - 1]
    d = add_mas(raw)
    all_sigs = detect_combo(d, cross_only=False)
    crosses = detect_combo(d, cross_only=True)
    assert len(crosses) >= 1
    assert all(s.crossed_200 for s in crosses)
    assert len(crosses) <= len(all_sigs)


def test_below_ma200_is_not_a_signal() -> None:
    n = 250
    close = np.full(n, 100.0)
    for i in range(1, n):
        close[i] = close[i - 1] - 0.05
    d = add_mas(
        {
            "t": np.arange(n, dtype=np.int64) * 60_000,
            "o": close,
            "h": close + 0.02,
            "l": close - 0.02,
            "c": close,
            "v": np.ones(n) * 10,
        }
    )
    assert detect_combo(d) == []


def test_default_date_uses_yesterday_before_2am() -> None:
    from datetime import datetime, timedelta, timezone

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from binance_1m_bull import default_date, day_window_ms

    tz = timezone(timedelta(hours=8))
    early = datetime(2026, 8, 25, 0, 58, tzinfo=tz)
    assert default_date(early) == "2026-08-24"
    late = datetime(2026, 8, 24, 17, 0, tzinfo=tz)
    assert default_date(late) == "2026-08-24"
    lo, hi = day_window_ms("2026-08-24")
    assert hi - lo == 24 * 60 * 60 * 1000


def test_is_usdt_um_perp() -> None:
    from nq.binance import is_usdt_um_perp

    ok = {
        "symbol": "BTCUSDT",
        "quoteAsset": "USDT",
        "marginAsset": "USDT",
        "status": "TRADING",
        "contractType": "PERPETUAL",
        "underlyingType": "COIN",
    }
    assert is_usdt_um_perp(ok)
    assert not is_usdt_um_perp({**ok, "quoteAsset": "USDC"})
    assert not is_usdt_um_perp({**ok, "marginAsset": "BTC"})
    assert not is_usdt_um_perp({**ok, "contractType": "CURRENT_QUARTER"})
    assert not is_usdt_um_perp({**ok, "underlyingType": "EQUITY"})
    assert not is_usdt_um_perp({**ok, "symbol": "USDCUSDT"})
    assert not is_usdt_um_perp({**ok, "status": "BREAK"})


def main() -> int:
    test_sma()
    test_detects_first_stack_above_ma200()
    test_no_repeat_while_stack_holds()
    test_rearms_after_break()
    test_min_gap()
    test_forward_and_summarize()
    test_cross_only_keeps_ma200_reclaim()
    test_below_ma200_is_not_a_signal()
    test_default_date_uses_yesterday_before_2am()
    test_is_usdt_um_perp()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

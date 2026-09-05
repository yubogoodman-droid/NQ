#!/usr/bin/env python3
"""Synthetic tests for 幣安 1h 多頭爆發（不打幣安）。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backtest_binance_1h_burst import (  # noqa: E402
    drop_overlap,
    simulate_trade,
    summarize,
)
from watch_binance_1h_burst import (  # noqa: E402
    MA_PERIODS,
    bars_from_raw,
    burst_at,
    drop_unclosed,
    find_bursts,
    format_burst,
    indicators,
    is_bull_align,
    key_of,
    sma,
)


def _uptrend(n: int = 240, start: float = 100.0, step: float = 0.12) -> dict:
    c = start + step * np.arange(n, dtype=float)
    o = c - 0.15
    h = c + 0.20
    l = c - 0.25
    v = np.full(n, 100.0)
    t = np.arange(n, dtype=np.int64) * 3_600_000
    return {"t": t, "o": o, "h": h, "l": l, "c": c, "v": v}


def test_sma() -> None:
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = sma(arr, 3)
    assert np.isnan(out[1])
    assert abs(out[2] - 2.0) < 1e-9
    assert abs(out[4] - 4.0) < 1e-9


def test_uptrend_is_bull_align() -> None:
    d = indicators(_uptrend())
    i = len(d["c"]) - 1
    mas = tuple(float(d[f"m{n}"][i]) for n in MA_PERIODS)
    assert is_bull_align(mas)
    assert mas[0] > mas[-1]


def test_burst_hits_on_double_volume() -> None:
    raw = _uptrend()
    raw["v"][-2] = 80.0
    raw["v"][-1] = 161.0  # 2.0125×
    d = indicators(raw)
    hit = burst_at(d, len(d["c"]) - 1, require_fan=False)
    assert hit is not None
    assert hit["vol_ratio"] > 2.0
    assert hit["close"] == raw["c"][-1]


def test_burst_rejects_exact_double() -> None:
    raw = _uptrend()
    raw["v"][-2] = 80.0
    raw["v"][-1] = 160.0
    d = indicators(raw)
    assert burst_at(d, len(d["c"]) - 1) is None


def test_burst_rejects_low_volume() -> None:
    raw = _uptrend()
    raw["v"][-2] = 100.0
    raw["v"][-1] = 150.0
    d = indicators(raw)
    assert burst_at(d, len(d["c"]) - 1) is None


def test_burst_rejects_bear_stack() -> None:
    raw = _uptrend()
    raw["c"] = raw["c"][::-1]
    raw["o"] = raw["c"] + 0.1
    raw["h"] = raw["c"] + 0.2
    raw["l"] = raw["c"] - 0.2
    raw["v"][-1] = 400.0
    d = indicators(raw)
    assert burst_at(d, len(d["c"]) - 1) is None


def test_burst_rejects_zero_prev_volume() -> None:
    raw = _uptrend()
    raw["v"][-2] = 0.0
    raw["v"][-1] = 500.0
    d = indicators(raw)
    assert burst_at(d, len(d["c"]) - 1) is None


def test_rejects_far_from_ma25() -> None:
    raw = _uptrend()
    raw["v"][-1] = 250.0
    raw["c"][-1] = raw["c"][-1] * 1.05
    raw["h"][-1] = raw["c"][-1]
    d = indicators(raw)
    i = len(d["c"]) - 1
    assert burst_at(d, i) is None
    assert burst_at(d, i, max_ext_ma25=0.08, require_fan=False) is not None
    assert burst_at(d, i) is None or d["c"][i] / d["m25"][i] - 1 > 0.015


def test_accepts_bnb_like_ma25() -> None:
    raw = _uptrend()
    raw["v"][-1] = 250.0
    d = indicators(raw)
    i = len(d["c"]) - 1
    hit = burst_at(d, i, require_fan=False)
    assert hit is not None
    assert 0 <= hit["ext_ma25"] <= 0.015


def _fan_bars(n: int = 240, grind: float = 1.00015, accel: float = 1.0012, acc_bars: int = 10) -> dict:
    """Slow grind then accelerate so short MAs open like BNB."""
    c = np.empty(n, dtype=float)
    c[0] = 100.0
    for i in range(1, n - acc_bars):
        c[i] = c[i - 1] * grind
    for i in range(n - acc_bars, n):
        c[i] = c[i - 1] * accel
    o = c - 0.05
    return {
        "t": np.arange(n, dtype=np.int64) * 3_600_000,
        "o": o,
        "h": c + 0.08,
        "l": c - 0.08,
        "c": c,
        "v": np.full(n, 100.0),
    }


def test_bnb_style_fan_passes() -> None:
    raw = _fan_bars()
    raw["v"][-1] = 250.0
    d = indicators(raw)
    hit = burst_at(d, len(d["c"]) - 1)
    assert hit is not None
    assert 0.002 <= hit["short_fan"] <= 0.008
    assert hit["fan_delta"] is not None and hit["fan_delta"] >= 0.001


def test_tight_stack_not_fan() -> None:
    raw = _uptrend(step=0.02)
    raw["v"][-1] = 250.0
    d = indicators(raw)
    assert burst_at(d, len(d["c"]) - 1) is None


def test_rejects_red_bar() -> None:
    raw = _uptrend()
    raw["o"][-1] = raw["c"][-1] + 0.5
    raw["v"][-1] = 250.0
    d = indicators(raw)
    i = len(d["c"]) - 1
    assert burst_at(d, i) is None
    assert burst_at(d, i, green_only=False, require_fan=False) is not None


def test_burst_rejects_early_bar() -> None:
    d = indicators(_uptrend(n=80))
    assert burst_at(d, 10) is None
    assert burst_at(d, 0) is None


def test_drop_unclosed() -> None:
    raw = [[0, 1, 1, 1, 1, 1], [3_600_000, 1, 1, 1, 1, 1]]
    assert len(drop_unclosed(raw, now_ms=3_600_000 + 10)) == 1
    assert len(drop_unclosed(raw, now_ms=3_600_000 + 3_600_000)) == 2


def test_bars_from_raw_needs_ma200() -> None:
    short = [[i * 3_600_000, 1, 1, 1, 1, 1] for i in range(50)]
    assert bars_from_raw(short) is None
    long = [[i * 3_600_000, 1, 2, 0.5, 1.2, 10] for i in range(210)]
    d = bars_from_raw(long)
    assert d is not None
    assert len(d["c"]) == 210


def test_find_bursts_window() -> None:
    raw = _uptrend()
    raw["v"][-3] = 80.0
    raw["v"][-2] = 170.0
    raw["v"][-1] = 90.0
    d = indicators(raw)
    n = len(d["c"])
    hits = find_bursts(d, n - 3, n - 1, require_fan=False)
    assert len(hits) == 1
    assert hits[0]["i"] == n - 2


def test_simulate_hits_target() -> None:
    raw = _uptrend()
    raw["v"][-20] = 80.0
    raw["v"][-19] = 200.0
    d = indicators(raw)
    i = len(d["c"]) - 19
    hit = burst_at(d, i, require_fan=False)
    assert hit is not None
    # next bars keep climbing; force a wide stop so 2R is reachable
    d["l"][i] = d["c"][i] * 0.99
    t = simulate_trade(d, hit, target_r=2.0, time_bars=8, min_risk_pct=0.005)
    assert t.reason in ("target", "time")
    assert t.entry == float(d["c"][i])


def test_simulate_hits_stop() -> None:
    raw = _uptrend()
    raw["v"][-12] = 80.0
    raw["v"][-11] = 200.0
    d = indicators(raw)
    i = len(d["c"]) - 11
    hit = burst_at(d, i, require_fan=False)
    assert hit is not None
    d["l"][i] = d["c"][i] * 0.99
    d["l"][i + 1] = d["c"][i] * 0.98
    d["h"][i + 1] = d["c"][i] * 1.001
    t = simulate_trade(d, hit, target_r=2.0, time_bars=8, min_risk_pct=0.005)
    assert t.reason == "stop"
    assert t.pnl_pct < 0


def test_drop_overlap_keeps_first() -> None:
    raw = _uptrend()
    d = indicators(raw)
    a = simulate_trade(
        d,
        {"i": 210, "vol_ratio": 3.0, "close": float(d["c"][210]), "open": float(d["o"][210])},
        time_bars=4,
    )
    b = simulate_trade(
        d,
        {"i": 212, "vol_ratio": 3.0, "close": float(d["c"][212]), "open": float(d["o"][212])},
        time_bars=4,
    )
    a.symbol = b.symbol = "BTCUSDT"
    kept = drop_overlap([b, a])
    assert len(kept) == 1
    assert kept[0].entry_idx == 210


def test_summarize_empty() -> None:
    s = summarize([])
    assert s["count"] == 0
    assert s["win_rate"] == 0.0


def test_key_and_format() -> None:
    raw = _uptrend()
    raw["v"][-1] = 250.0
    d = indicators(raw)
    hit = burst_at(d, len(d["c"]) - 1, require_fan=False)
    ev = {"symbol": "BTCUSDT", "d": d, **hit}
    assert key_of(ev) == f"BTCUSDT:{int(d['t'][-1])}"
    msg = format_burst(ev)
    assert "BTCUSDT" in msg
    assert "多頭爆發" in msg
    assert "MA7" in msg and "MA200" in msg
    assert "MA25" in msg


def main() -> int:
    tests = [
        test_sma,
        test_uptrend_is_bull_align,
        test_burst_hits_on_double_volume,
        test_burst_rejects_exact_double,
        test_burst_rejects_low_volume,
        test_burst_rejects_bear_stack,
        test_burst_rejects_zero_prev_volume,
        test_rejects_far_from_ma25,
        test_accepts_bnb_like_ma25,
        test_bnb_style_fan_passes,
        test_tight_stack_not_fan,
        test_rejects_red_bar,
        test_burst_rejects_early_bar,
        test_drop_unclosed,
        test_bars_from_raw_needs_ma200,
        test_find_bursts_window,
        test_simulate_hits_target,
        test_simulate_hits_stop,
        test_drop_overlap_keeps_first,
        test_summarize_empty,
        test_key_and_format,
    ]
    for fn in tests:
        fn()
        print("ok", fn.__name__)
    print(f"{len(tests)} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

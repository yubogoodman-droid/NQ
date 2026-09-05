#!/usr/bin/env python3
"""Synthetic tests for 幣安 1h 多頭爆發（不打幣安）。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from watch_binance_1h_burst import (  # noqa: E402
    MA_PERIODS,
    bars_from_raw,
    burst_at,
    drop_unclosed,
    format_burst,
    indicators,
    is_bull_align,
    key_of,
    sma,
)


def _uptrend(n: int = 240, start: float = 100.0, step: float = 0.4) -> dict:
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
    hit = burst_at(d, len(d["c"]) - 1)
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


def test_key_and_format() -> None:
    raw = _uptrend()
    raw["v"][-1] = 250.0
    d = indicators(raw)
    hit = burst_at(d, len(d["c"]) - 1)
    ev = {"symbol": "BTCUSDT", "d": d, **hit}
    assert key_of(ev) == f"BTCUSDT:{int(d['t'][-1])}"
    msg = format_burst(ev)
    assert "BTCUSDT" in msg
    assert "多頭爆發" in msg
    assert "MA7" in msg and "MA200" in msg


def main() -> int:
    tests = [
        test_sma,
        test_uptrend_is_bull_align,
        test_burst_hits_on_double_volume,
        test_burst_rejects_exact_double,
        test_burst_rejects_low_volume,
        test_burst_rejects_bear_stack,
        test_burst_rejects_zero_prev_volume,
        test_burst_rejects_early_bar,
        test_drop_unclosed,
        test_bars_from_raw_needs_ma200,
        test_key_and_format,
    ]
    for fn in tests:
        fn()
        print("ok", fn.__name__)
    print(f"{len(tests)} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""15m MA5/20/99 多頭排列、剛站上 MA200、均線距離像 ETH 圖。"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nq.ma_align import (  # noqa: E402
    MIN_BARS,
    add_indicators,
    detect_signals,
    is_above_200,
    is_stacked,
    signal_at,
)


def _bars(close: np.ndarray) -> dict:
    close = np.asarray(close, float)
    n = len(close)
    o = np.concatenate([[close[0]], close[:-1]])
    rng = np.maximum(np.abs(close - o), close * 0.001)
    h = np.maximum(o, close) + rng * 0.2
    l = np.minimum(o, close) - rng * 0.2
    t0 = int(datetime(2026, 9, 1, 8, 0).timestamp() * 1000)
    t = t0 + np.arange(n, dtype=np.int64) * 900_000
    return {"t": t, "o": o, "h": h, "l": l, "c": close, "v": np.full(n, 1000.0)}


def _blank(n: int = 240) -> dict:
    return add_indicators(_bars(np.full(n, 100.0)))


def _eth_like(d: dict, i: int, *, above: bool) -> None:
    """對齊 ETH 9/3 20:00：5>20>99，四條黏在 0.53%，剛貼過 200。"""
    d["m5"][i], d["m20"][i], d["m99"][i], d["m200"][i] = 100.52, 100.35, 100.05, 100.58
    d["c"][i] = 100.68 if above else 100.40


def test_just_stood_on_200_hits() -> None:
    d = _blank()
    i = 220
    _eth_like(d, i - 1, above=False)
    _eth_like(d, i, above=True)
    sig = signal_at(d, i)
    assert sig is not None
    assert sig.ribbon <= 0.008
    assert sig.gap99 <= 0.007
    assert sig.ext <= 0.010


def test_already_above_200_skips() -> None:
    d = _blank()
    i = 220
    _eth_like(d, i - 1, above=True)
    _eth_like(d, i, above=True)
    assert signal_at(d, i) is None
    assert is_above_200(d, i)


def test_wide_ribbon_skips() -> None:
    """均線散開，不像截圖那種黏帶。"""
    d = _blank()
    i = 220
    d["c"][i - 1], d["m200"][i - 1] = 99.5, 100.0
    d["c"][i] = 101.0
    d["m5"][i], d["m20"][i], d["m99"][i], d["m200"][i] = 102.5, 101.2, 98.0, 100.0
    assert is_stacked(d, i)
    assert signal_at(d, i) is None


def test_99_far_from_200_skips() -> None:
    d = _blank()
    i = 220
    d["c"][i - 1], d["m200"][i - 1] = 99.5, 100.0
    d["c"][i] = 100.4
    d["m5"][i], d["m20"][i], d["m99"][i], d["m200"][i] = 100.3, 100.2, 98.5, 100.0
    assert signal_at(d, i) is None


def test_below_200_skips() -> None:
    d = _blank()
    i = 220
    _eth_like(d, i, above=False)
    assert signal_at(d, i) is None


def test_not_stacked_skips() -> None:
    d = _blank()
    i = 220
    d["c"][i - 1] = 99.8
    d["m200"][i - 1] = 100.1
    d["c"][i] = 100.5
    d["m5"][i], d["m20"][i], d["m99"][i], d["m200"][i] = 100.2, 101.5, 100.6, 100.1
    assert signal_at(d, i) is None
    assert not is_stacked(d, i)


def test_detect_only_cross_bar() -> None:
    d = _blank(260)
    _eth_like(d, 219, above=False)
    for i in range(220, 230):
        _eth_like(d, i, above=True)
    hits = detect_signals(d)
    assert len(hits) == 1
    assert hits[0].idx == 220


def test_history_required() -> None:
    d = add_indicators(_bars(np.full(50, 100.0)))
    assert signal_at(d, 40) is None
    assert MIN_BARS >= 200


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"ok  {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    if failed:
        raise SystemExit(1)
    print(f"{len(tests)} passed")

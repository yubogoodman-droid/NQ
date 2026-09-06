#!/usr/bin/env python3
"""15m MA5/20/99 多頭排列站上 MA200：合成 K 線測試（不連網）。"""
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
    is_aligned,
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


def test_first_align_hits() -> None:
    d = _blank()
    i = 220
    d["c"][i] = 102.0
    d["m5"][i], d["m20"][i], d["m99"][i], d["m200"][i] = 101.6, 101.2, 100.6, 100.1
    d["c"][i - 1] = 99.8
    d["m5"][i - 1], d["m20"][i - 1], d["m99"][i - 1], d["m200"][i - 1] = 100.4, 100.3, 100.2, 100.2
    sig = signal_at(d, i)
    assert sig is not None
    assert sig.ma5 > sig.ma20 > sig.ma99 > sig.ma200
    assert sig.close > sig.ma200


def test_already_aligned_skips() -> None:
    d = _blank()
    i = 220
    for k in (i - 1, i):
        d["c"][k] = 102.0
        d["m5"][k], d["m20"][k], d["m99"][k], d["m200"][k] = 101.6, 101.2, 100.6, 100.1
    assert signal_at(d, i) is None
    assert is_aligned(d, i)


def test_below_200_skips() -> None:
    d = _blank()
    i = 220
    d["c"][i] = 99.5
    d["m5"][i], d["m20"][i], d["m99"][i], d["m200"][i] = 101.6, 101.2, 100.6, 100.1
    assert signal_at(d, i) is None


def test_not_stacked_skips() -> None:
    d = _blank()
    i = 220
    d["c"][i] = 102.0
    d["m5"][i], d["m20"][i], d["m99"][i], d["m200"][i] = 100.2, 101.5, 100.6, 100.1
    assert signal_at(d, i) is None


def test_detect_only_first_bar() -> None:
    d = _blank(260)
    for i in range(220, 230):
        d["c"][i] = 102.0
        d["m5"][i], d["m20"][i], d["m99"][i], d["m200"][i] = 101.6, 101.2, 100.6, 100.1
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

#!/usr/bin/env python3
"""15m 壓縮擴張：合成 K 線測試（不連網）。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scan_binance_15m_expansion import (  # noqa: E402
    collapse_hits,
    detect_expansion,
    indicators,
    rsi_sma,
    sma,
)


def _bars(n: int, close: np.ndarray, vol: np.ndarray | None = None) -> dict:
    close = np.asarray(close, float)
    rng = np.maximum(np.abs(np.diff(close, prepend=close[0])) * 0.35, close * 0.001)
    o = np.concatenate([[close[0]], close[:-1]])
    h = np.maximum(o, close) + rng * 0.25
    l = np.minimum(o, close) - rng * 0.25
    v = vol if vol is not None else np.full(n, 1000.0)
    t = np.arange(n, dtype=np.int64) * 900_000
    return {"t": t, "o": o, "h": h, "l": l, "c": close, "v": v}


def test_sma() -> None:
    out = sma(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 3)
    assert np.isnan(out[1])
    assert abs(out[2] - 2.0) < 1e-9
    assert abs(out[4] - 4.0) < 1e-9


def test_rsi_sma_all_up() -> None:
    c = np.linspace(1.0, 2.0, 40)
    r = rsi_sma(c, 6)
    assert r[-1] > 90


def _tight(n: int, px: float, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    wobble = rng.uniform(-0.0018, 0.0018, n)
    return px * (1.0 + np.cumsum(wobble) * 0) * (1.0 + wobble)


def test_detect_fil_like_vertical() -> None:
    """左邊橫盤，右邊連續大陽線。"""
    n0 = 90
    close = np.concatenate([_tight(n0, 1.30), 1.30 * (1.022 ** np.arange(1, 9))])
    vol = np.concatenate([np.full(n0, 1_000.0), np.full(8, 12_000.0)])
    d = indicators(_bars(len(close), close, vol))
    hits = detect_expansion(d)
    assert hits, "FIL 那種連陽應該命中"
    last = hits[-1]
    assert last["kind"] == "vertical"
    assert last["move"] >= 0.07
    assert last["cons"] >= 5


def test_detect_pippin_like_stair() -> None:
    """一段一段往上墊，中間夾陰線。"""
    n0 = 90
    px = 0.29
    stair = []
    for i in range(24):
        if i % 5 == 4:
            px *= 0.994
        else:
            px *= 1.0085
        stair.append(px)
    close = np.concatenate([_tight(n0, 0.29, seed=2), np.array(stair)])
    vol = np.concatenate([np.full(n0, 2_000.0), np.full(24, 5_500.0)])
    d = indicators(_bars(len(close), close, vol))
    hits = detect_expansion(d)
    assert hits, "PIPPIN 那種墊高應該命中"
    assert hits[-1]["move"] >= 0.07
    assert hits[-1]["green_ratio"] >= 0.55


def test_detect_crcl_like_volume_spike() -> None:
    n0 = 90
    body = 71.4 * (1.012 ** np.arange(1, 13))
    close = np.concatenate([_tight(n0, 71.4, seed=3), body])
    vol = np.concatenate([np.full(n0, 8_000.0), np.full(12, 160_000.0)])
    d = indicators(_bars(len(close), close, vol))
    hits = detect_expansion(d)
    assert hits, "CRCL 那種放量噴出應該命中"
    assert hits[-1]["vol_ratio"] >= 1.5


def test_chop_does_not_hit() -> None:
    rng = np.random.default_rng(7)
    close = 100.0 * (1.0 + rng.uniform(-0.004, 0.004, 160))
    d = indicators(_bars(len(close), close))
    assert detect_expansion(d) == []


def test_slow_grind_does_not_hit() -> None:
    """25 小時才走 4%，不該當成噴出。"""
    close = 50.0 * (1.00025 ** np.arange(160))
    d = indicators(_bars(len(close), close, np.full(160, 3000.0)))
    assert detect_expansion(d) == []


def test_collapse_keeps_best_of_run() -> None:
    hits = [
        {"i": 10, "score": 1.0},
        {"i": 11, "score": 3.0},
        {"i": 12, "score": 2.0},
        {"i": 20, "score": 4.0},
    ]
    out = collapse_hits(hits)
    assert [h["i"] for h in out] == [11, 20]
    assert out[0]["score"] == 3.0


def main() -> int:
    test_sma()
    test_rsi_sma_all_up()
    test_detect_fil_like_vertical()
    test_detect_pippin_like_stair()
    test_detect_crcl_like_volume_spike()
    test_chop_does_not_hit()
    test_slow_grind_does_not_hit()
    test_collapse_keeps_best_of_run()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

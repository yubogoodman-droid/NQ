#!/usr/bin/env python3
"""幣安 5m W 底掃描（不打網路）。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scan_binance_w_bottom import (  # noqa: E402
    detect_w_bottoms,
    draw_hit_png,
    fmt_price,
    swing_lows,
)


def _series() -> tuple[list[float], ...]:
    """合成一組近似 UAI 的 W 底：先急跌、雙底、再突破頸線。"""
    closes = [round(0.62 * (0.996 ** i), 5) for i in range(36)]
    # 第一底 → 頸線 → 第二底 → 突破
    body = [
        0.530, 0.518, 0.505, 0.495, 0.492, 0.505, 0.515, 0.525, 0.532, 0.531,
        0.522, 0.512, 0.504, 0.500, 0.498, 0.508, 0.520, 0.532, 0.545, 0.552,
        0.561, 0.570, 0.582, 0.595, 0.610, 0.625, 0.640, 0.655, 0.668, 0.680,
    ]
    closes.extend(body)
    while len(closes) < 90:
        closes.append(round(closes[-1] * 1.002, 5))
    n = len(closes)
    opens = [closes[0]] + closes[:-1]
    highs = [max(o, c) * 1.004 for o, c in zip(opens, closes)]
    lows = [min(o, c) * 0.998 for o, c in zip(opens, closes)]
    i1, neck_i, i2 = 40, 45, 50
    lows[i1] = 0.4885
    closes[i1] = 0.492
    highs[i1] = 0.500
    for k in range(i1 - 3, i1 + 4):
        if k != i1:
            lows[k] = max(lows[k], 0.495)
    highs[neck_i] = 0.5353
    closes[neck_i] = 0.531
    lows[i2] = 0.4963
    closes[i2] = 0.500
    highs[i2] = 0.508
    for k in range(i2 - 3, i2 + 4):
        if k != i2:
            lows[k] = max(lows[k], 0.502)
    quote = [1_000_000.0] * n
    quote[i2 + 6] = 4_000_000.0
    times = [1_788_000_000_000 + i * 300_000 for i in range(n)]
    return opens, highs, lows, closes, quote, times


def test_swing_lows_finds_two_bottoms() -> None:
    _o, _h, lows, *_ = _series()
    found = swing_lows(lows)
    assert any(abs(lows[i] - 0.4885) < 1e-9 for i in found)
    assert any(abs(lows[i] - 0.4963) < 1e-9 for i in found)


def test_detect_w_bottom_like_uai() -> None:
    hits = detect_w_bottoms(*_series(), symbol="UAIUSDT", volume24=1e8)
    assert hits, "應偵測到至少一個 W 底"
    best = max(hits, key=lambda h: h.score)
    assert abs(best.bottom1 - 0.4885) < 1e-6
    assert abs(best.bottom2 - 0.4963) < 1e-6
    assert best.neck > best.bottom2
    assert 7 <= best.depth_pct <= 18
    assert best.status in {"突破仍有效", "已延伸"}
    assert best.b is not None


def test_fmt_price() -> None:
    assert fmt_price(123.456) == "123.46"
    assert fmt_price(0.4885) == "0.48850"


def test_draw_hit_png(tmp_path: Path | None = None) -> None:
    hits = detect_w_bottoms(*_series(), symbol="UAIUSDT", volume24=1e8)
    best = max(hits, key=lambda h: h.score)
    out = (tmp_path or Path("/tmp")) / "w_bottom_test.png"
    draw_hit_png(best, _series(), out, "UAIUSDT 5m W底")
    assert out.exists()
    assert out.stat().st_size > 1000


if __name__ == "__main__":
    test_swing_lows_finds_two_bottoms()
    test_detect_w_bottom_like_uai()
    test_fmt_price()
    test_draw_hit_png()
    print("ok")

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
    kline_count,
    like_pct_from_score,
    ma_hold,
    merge_klines,
    swing_lows,
    uai_like_score,
)


def _paint_w(closes: list[float], start: int) -> tuple[int, int, int]:
    """在 start 之後接急殺 + 雙底 + 突破，回傳 L1 / 頸線 / L2 索引。"""
    dump = [round(0.60 * (0.985 ** i), 5) for i in range(12)]
    body = [
        0.530, 0.518, 0.505, 0.495, 0.492, 0.505, 0.515, 0.525, 0.532, 0.531,
        0.522, 0.512, 0.504, 0.500, 0.498, 0.508, 0.520, 0.532, 0.545, 0.552,
        0.561, 0.570, 0.582, 0.595, 0.610, 0.625, 0.640, 0.655, 0.668, 0.680,
    ]
    closes.extend(dump)
    closes.extend(body)
    i1, neck_i, i2 = start + 16, start + 21, start + 26
    return i1, neck_i, i2


def _finalize(closes: list[float], i1: int, neck_i: int, i2: int) -> tuple[list[float], ...]:
    while len(closes) < i2 + 40:
        closes.append(round(closes[-1] * 1.002, 5))
    n = len(closes)
    opens = [closes[0]] + closes[:-1]
    highs = [max(o, c) * 1.004 for o, c in zip(opens, closes)]
    lows = [min(o, c) * 0.998 for o, c in zip(opens, closes)]
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


def _series() -> tuple[list[float], ...]:
    """長時間在 0.49 附近盤，急殺後雙底踩回 MA200。"""
    closes = [0.490] * 220
    i1, neck_i, i2 = _paint_w(closes, 220)
    return _finalize(closes, i1, neck_i, i2)


def _series_no_ma_floor() -> tuple[list[float], ...]:
    """均線在 0.70，雙底懸在 0.49，不應算有支撐。"""
    closes = [0.70] * 220
    i1, neck_i, i2 = _paint_w(closes, 220)
    return _finalize(closes, i1, neck_i, i2)


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
    assert best.like_pct >= 50
    assert best.bottom2 >= best.bottom1 * 0.995
    assert best.ma_support in {"MA200", "MA120", "MA99"}


def test_rejects_w_without_ma_support() -> None:
    hits = detect_w_bottoms(*_series_no_ma_floor(), symbol="FAKEUSDT", volume24=1e8)
    assert not hits


def test_ma_hold() -> None:
    assert ma_hold(0.4885, 0.5062, 0.4924, 0.025)
    assert not ma_hold(0.2249, 0.2363, 0.3253, 0.025)


def test_uai_like_prefers_compact_higher_low() -> None:
    close = uai_like_score(
        sep=13, depth=0.087, hl=0.016, dump=0.21, breakout_bars=2, ext=0.25, now=0.12, volx=2.5, age_h=3
    )
    wide_lower = uai_like_score(
        sep=36, depth=0.12, hl=-0.025, dump=0.09, breakout_bars=None, ext=0.0, now=-0.05, volx=None, age_h=6
    )
    assert like_pct_from_score(close) >= 80
    assert like_pct_from_score(wide_lower) < 40


def test_kline_count_week() -> None:
    assert kline_count(7) == 7 * 24 * 12 + 220


def test_merge_klines_dedupes() -> None:
    a = [[100, "1"], [200, "2"]]
    b = [[200, "2b"], [300, "3"]]
    merged = merge_klines(a, b)
    assert [row[0] for row in merged] == [100, 200, 300]
    assert merged[1][1] == "2"


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
    test_rejects_w_without_ma_support()
    test_ma_hold()
    test_uai_like_prefers_compact_higher_low()
    test_kline_count_week()
    test_merge_klines_dedupes()
    test_fmt_price()
    test_draw_hit_png()
    print("ok")

#!/usr/bin/env python3
"""15 分爆量穿六均線。合成資料，不打幣安。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from binance_ma_burst import MA_PERIODS, VOL_WINDOW, burst_at, find_bursts, sma  # noqa: E402


def flat(n: int = 260, price: float = 10.0, vol: float = 100.0):
    c = np.full(n, price)
    o = np.full(n, price)
    v = np.full(n, vol)
    return o, c, v


def test_sma() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    got = sma(x, 3)
    assert np.isnan(got[0]) and np.isnan(got[1])
    assert got[2] == 2.0 and abs(got[4] - 4.0) < 1e-9


def test_one_bar_clears_all_six_with_volume() -> None:
    o, c, v = flat()
    i = len(c) - 1
    o[i] = 9.9
    c[i] = 10.8
    v[i] = 100.0 * VOL_WINDOW + 50  # 大於前 10 根合計
    hits = find_bursts(o, c, v)
    assert [h["i"] for h in hits] == [i]
    hit = hits[0]
    assert hit["volume"] > hit["prior_volume"]
    assert set(hit["mas"]) == set(MA_PERIODS)
    assert all(hit["open"] <= hit["mas"][n] < hit["close"] for n in MA_PERIODS)


def test_quiet_cross_is_not_a_burst() -> None:
    """均量的幾倍還不夠。爆量是這根比前 10 根加總還大。"""
    o, c, v = flat()
    i = len(c) - 1
    o[i] = 9.9
    c[i] = 10.8
    v[i] = 100.0 * VOL_WINDOW - 50
    assert find_bursts(o, c, v) == []


def test_follow_through_above_the_ribbon_is_not_a_cross() -> None:
    """大陽開盤已經在六條之上，是延伸，不是這根在突破。"""
    o, c, v = flat()
    i = len(c) - 1
    o[i] = 10.4
    c[i] = 12.0
    v[i] = 100.0 * 30
    assert find_bursts(o, c, v) == []


def test_must_clear_every_ma() -> None:
    # 長均還在上面，這根只穿過短均，不算一次突破六條。
    o, c, v = flat(n=260, price=12.0)
    o[-31:] = 10.0
    c[-31:-1] = 10.0
    i = len(c) - 1
    o[i] = 9.8
    c[i] = 10.5
    v[i] = 100.0 * 40
    ma = {n: sma(c, n) for n in MA_PERIODS}
    assert c[i] > ma[7][i]
    assert c[i] < ma[200][i]
    assert burst_at(o, c, v, ma, i) is None


def test_only_the_crossing_bar_fires() -> None:
    o, c, v = flat(n=280)
    i = 250
    o[i] = 9.9
    c[i] = 10.8
    v[i] = 100.0 * VOL_WINDOW + 200
    # 下一根沿高檔繼續放量，開盤已在均線上
    o[i + 1] = 10.8
    c[i + 1] = 12.5
    v[i + 1] = 100.0 * 40
    hits = find_bursts(o, c, v)
    assert [h["i"] for h in hits] == [i]


def test_needs_warmup() -> None:
    o, c, v = flat(n=VOL_WINDOW + 5)
    i = len(c) - 1
    o[i] = 9.0
    c[i] = 12.0
    v[i] = 100.0 * 50
    assert find_bursts(o, c, v) == []


def main() -> int:
    test_sma()
    test_one_bar_clears_all_six_with_volume()
    test_quiet_cross_is_not_a_burst()
    test_follow_through_above_the_ribbon_is_not_a_cross()
    test_must_clear_every_ma()
    test_only_the_crossing_bar_fires()
    test_needs_warmup()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

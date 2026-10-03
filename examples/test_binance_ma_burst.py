#!/usr/bin/env python3
"""15 分爆量穿六均線。合成資料，不打幣安。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from binance_ma_burst import (  # noqa: E402
    HOUR_MS,
    INTERVAL_MS,
    MA_PERIODS,
    VOL_MULT,
    above_hour_ma200,
    burst_at,
    find_bursts,
    hour_ma200_for_bars,
    hour_view,
    path_after,
    signal_hour_index,
    sma,
)


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
    v[i] = 100.0 * VOL_MULT
    hits = find_bursts(o, c, v)
    assert [h["i"] for h in hits] == [i]
    hit = hits[0]
    assert hit["vol_ratio"] == VOL_MULT
    assert hit["prior_volume"] == 100.0
    assert set(hit["mas"]) == set(MA_PERIODS)
    assert all(hit["open"] <= hit["mas"][n] < hit["close"] for n in MA_PERIODS)


def test_quiet_cross_is_not_a_burst() -> None:
    o, c, v = flat()
    i = len(c) - 1
    o[i] = 9.9
    c[i] = 10.8
    v[i] = 100.0 * VOL_MULT - 1
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
    v[i] = 100.0 * VOL_MULT
    # 下一根沿高檔繼續放量，開盤已在均線上
    o[i + 1] = 10.8
    c[i + 1] = 12.5
    v[i + 1] = 100.0 * 40
    hits = find_bursts(o, c, v)
    assert [h["i"] for h in hits] == [i]


def test_path_after() -> None:
    d = {
        "c": np.array([10.0, 10.2, 11.0, 10.5, 11.0]),
        "h": np.array([10.1, 10.4, 12.0, 10.8, 11.2]),
        "l": np.array([9.8, 9.9, 10.1, 9.0, 10.4]),
    }
    got = path_after(d, 0, 4)
    assert got is not None
    assert got["bars"] == 4
    assert abs(got["ret"] - 0.10) < 1e-9
    assert abs(got["mfe"] - 0.20) < 1e-9
    assert abs(got["mae"] - (-0.10)) < 1e-9
    assert path_after(d, 4, 4) is None


def test_hour_ma200_uses_last_closed_hour() -> None:
    n = 210
    hour_open = np.arange(n, dtype=np.int64) * HOUR_MS
    hour_close = np.full(n, 50.0)
    hour_close[-1] = 80.0
    # 收盤正好卡在整點：剛走完的是最後那根小時 K，MA200 被 80 拉高
    on_hour = np.array([hour_open[-1] + HOUR_MS - INTERVAL_MS], dtype=np.int64)
    got = hour_ma200_for_bars(on_hour, hour_open, hour_close)
    expect = (50.0 * 199 + 80.0) / 200
    assert abs(got[0] - expect) < 1e-9
    # 收盤在整點後 15 分：最新走完的仍是同一根，不能用還沒走完的下一根
    inside = np.array([hour_open[-1] + HOUR_MS], dtype=np.int64)
    got2 = hour_ma200_for_bars(inside, hour_open, hour_close)
    assert abs(got2[0] - expect) < 1e-9
    assert above_hour_ma200(expect + 0.01, got[0])
    assert not above_hour_ma200(expect, got[0])
    assert not above_hour_ma200(1.0, float("nan"))


def test_signal_hour_index_marks_the_hour_that_holds_the_bar() -> None:
    hour_open = np.arange(5, dtype=np.int64) * HOUR_MS
    # 22:15 那根 15 分（開盤在整點後 15 分）仍落在同一根小時 K
    inside = int(hour_open[3] + INTERVAL_MS)
    assert signal_hour_index(inside, hour_open) == 3
    # 訊號落在還沒走完、資料裡還沒有的那根小時，標已經收盤的前一根
    forming = int(hour_open[-1] + HOUR_MS)
    assert signal_hour_index(forming, hour_open) == 4
    view = hour_view(hour_open, int(hour_open[3]), before=2, after=1)
    assert view == (1, 5, 3)
    assert signal_hour_index(0, np.array([], dtype=np.int64)) == -1
    assert hour_view(np.array([], dtype=np.int64), 0) is None


def test_needs_warmup() -> None:
    o, c, v = flat(n=30)
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
    test_path_after()
    test_hour_ma200_uses_last_closed_hour()
    test_signal_hour_index_marks_the_hour_that_holds_the_bar()
    test_needs_warmup()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

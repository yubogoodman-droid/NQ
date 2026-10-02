#!/usr/bin/env python3
"""ORCL 1m MA7/MA14 死亡交叉且破 MA25（不打網路）。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from watch_orcl_death_cross import detect_shorts, lead_bars, sma  # noqa: E402


def test_sma() -> None:
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = sma(arr, 3)
    assert np.isnan(out[1])
    assert abs(out[2] - 2.0) < 1e-9
    assert abs(out[4] - 4.0) < 1e-9


def _mas(close: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return sma(close, 7), sma(close, 14), sma(close, 25)


def test_death_and_below_fires() -> None:
    n = 80
    close = np.linspace(100.0, 110.0, n)
    # 最後幾根急殺，讓 MA7 下穿 MA14，收盤掉到 MA25 下
    close[-8:] = np.array([109.2, 108.4, 107.2, 105.8, 104.1, 102.4, 100.6, 99.0])
    m7, m14, m25 = _mas(close)
    hits = detect_shorts(close, m7, m14, m25, min_lead=3)
    assert hits, "急殺應出現死亡交叉且破 MA25"
    hit = hits[-1]
    assert close[hit.i] < m25[hit.i]
    assert m7[hit.i] < m14[hit.i]
    assert m7[hit.i - 1] >= m14[hit.i - 1]


def test_death_above_ma25_skipped() -> None:
    n = 60
    close = np.linspace(100.0, 108.0, n)
    # 輕微回檔：7/14 交叉但價格仍在 MA25 上
    close[-4:] = np.array([107.85, 107.70, 107.55, 107.45])
    m7, m14, m25 = _mas(close)
    raw = detect_shorts(close, m7, m14, m25, min_lead=0)
    for h in raw:
        assert close[h.i] < m25[h.i]


def test_only_the_cross_bar() -> None:
    n = 80
    close = np.linspace(100.0, 110.0, n)
    close[-10:] = np.array([109.4, 108.6, 107.0, 105.2, 103.4, 102.0, 101.2, 100.8, 100.5, 100.3])
    m7, m14, m25 = _mas(close)
    hits = detect_shorts(close, m7, m14, m25, min_lead=0)
    assert hits
    # 交叉後持續 7<14 且收在 25 下，不應每根都發
    idxs = [h.i for h in hits]
    assert idxs == sorted(set(idxs))
    for i in idxs:
        assert m7[i - 1] >= m14[i - 1] and m7[i] < m14[i]


def test_min_lead_filters_flicker() -> None:
    close = np.array(
        [10.0] * 20
        + [10.2, 10.4, 10.6, 10.8, 11.0, 11.1, 11.2, 11.0, 10.4, 9.6, 9.0],
        dtype=float,
    )
    # 補到夠算 MA25
    close = np.concatenate([np.full(20, 10.0), close])
    m7, m14, m25 = _mas(close)
    loose = detect_shorts(close, m7, m14, m25, min_lead=0)
    tight = detect_shorts(close, m7, m14, m25, min_lead=20)
    assert len(tight) <= len(loose)


def test_require_cross_ma25() -> None:
    n = 80
    close = np.linspace(100.0, 110.0, n)
    close[-12:-6] = np.array([108.0, 106.5, 105.0, 103.5, 102.0, 101.0])
    close[-6:] = np.array([100.5, 100.2, 100.0, 99.8, 99.6, 99.4])
    m7, m14, m25 = _mas(close)
    any_below = detect_shorts(close, m7, m14, m25, min_lead=0, require_cross_ma25=False)
    same_bar = detect_shorts(close, m7, m14, m25, min_lead=0, require_cross_ma25=True)
    assert all(h.crossed_ma25 for h in same_bar)
    assert len(same_bar) <= len(any_below)


def test_lead_bars_counts_pre_cross() -> None:
    m7 = np.array([1.0, 2.0, 3.0, 4.0, 3.5, 2.0])
    m14 = np.array([1.5, 1.8, 2.5, 3.2, 3.4, 3.0])
    # i=5: 2.0 < 3.0 death; prior 3.5>=3.4 (1 bar), 4>=3.2, 3>=2.5, 2>=1.8 → 4 bars
    assert lead_bars(m7, m14, 5) == 4


def test_dump_like_screenshot() -> None:
    """高點後急殺：死亡交叉當根收盤跌破 MA25（對齊 10-02 22:49 那波）。"""
    n = 90
    close = np.linspace(140.0, 144.9, n - 8)
    dump = np.array([144.58, 144.44, 144.23, 144.10, 143.89, 143.64, 143.47, 143.41])
    close = np.concatenate([close, dump])
    m7, m14, m25 = _mas(close)
    hits = detect_shorts(close, m7, m14, m25, min_lead=5)
    assert hits, "這波急殺應抓得到"
    hit = hits[-1]
    assert hit.crossed_ma25 or close[hit.i] < m25[hit.i]
    assert m7[hit.i] < m14[hit.i]
    # 訊號發生在急殺段，不是還在創新高時
    assert hit.i >= n - 8


def main() -> int:
    test_sma()
    test_death_and_below_fires()
    test_death_above_ma25_skipped()
    test_only_the_cross_bar()
    test_min_lead_filters_flicker()
    test_require_cross_ma25()
    test_lead_bars_counts_pre_cross()
    test_dump_like_screenshot()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

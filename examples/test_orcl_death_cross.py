#!/usr/bin/env python3
"""ORCL 1m MA7/MA14 死亡交叉且破 MA25（不打網路）。"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from watch_orcl_death_cross import (  # noqa: E402
    ScanRow,
    ShortHit,
    cutoff_ms,
    day_bounds_ms,
    detect_shorts,
    filter_universe,
    forward_moves,
    in_window,
    latest_high_in_window,
    lead_bars,
    new_high_mask,
    orcl_like,
    pct_move,
    sma,
    taipei_day,
    write_html_report,
)


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


def test_filter_universe_keeps_orcl_drops_index() -> None:
    info = [
        {"symbol": "BTCUSDT", "quoteAsset": "USDT", "status": "TRADING", "contractType": "PERPETUAL", "underlyingType": "COIN"},
        {"symbol": "ORCLUSDT", "quoteAsset": "USDT", "status": "TRADING", "contractType": "TRADIFI_PERPETUAL", "underlyingType": "EQUITY"},
        {"symbol": "DEADUSDT", "quoteAsset": "USDT", "status": "TRADING", "contractType": "PERPETUAL", "underlyingType": "COIN"},
        {"symbol": "IDXUSDT", "quoteAsset": "USDT", "status": "TRADING", "contractType": "PERPETUAL", "underlyingType": "INDEX"},
        {"symbol": "ETHBTC", "quoteAsset": "BTC", "status": "TRADING", "contractType": "PERPETUAL", "underlyingType": "COIN"},
    ]
    tickers = {
        "BTCUSDT": {"quoteVolume": "10000000"},
        "ORCLUSDT": {"quoteVolume": "100"},
        "DEADUSDT": {"quoteVolume": "1"},
        "IDXUSDT": {"quoteVolume": "99999999"},
    }
    out = filter_universe(info, tickers, min_quote_vol=5_000_000, keep={"ORCLUSDT"})
    assert out == ["BTCUSDT", "ORCLUSDT"]
    all_usdt = filter_universe(info, tickers, min_quote_vol=0)
    assert "DEADUSDT" in all_usdt
    assert "IDXUSDT" not in all_usdt
    assert "ETHBTC" not in all_usdt


def test_day_window() -> None:
    start, end = day_bounds_ms("2026-10-02")
    # 22:49 台北 should be inside
    ts = int(datetime(2026, 10, 2, 22, 49, tzinfo=timezone(timedelta(hours=8))).timestamp() * 1000)
    assert in_window(ts, start, end)
    assert taipei_day(ts) == "2026-10-02"
    assert not in_window(start - 1, start, end)
    assert not in_window(end, start, end)
    s2, e2 = cutoff_ms(day="2026-10-02", hours=24)
    assert (s2, e2) == (start, end)


def test_forward_moves_dump() -> None:
    close = np.array([100.0, 99.0, 98.0, 97.0, 96.0], dtype=float)
    low = np.array([99.5, 98.5, 97.5, 96.5, 95.0], dtype=float)
    f15, f30, l15, l30 = forward_moves(close, low, 0, n15=2, n30=4)
    assert f15 is not None and abs(f15 - pct_move(100, 98)) < 1e-9
    assert l30 is not None and abs(l30 - pct_move(100, 95)) < 1e-9
    empty = forward_moves(close, low, 4, n15=2, n30=4)
    assert empty == (None, None, None, None)


def test_write_html_report(tmp_path=None) -> None:
    from pathlib import Path
    import tempfile

    hit = ShortHit(i=10, close=144.44, m7=144.63, m14=144.65, m25=144.55, lead=22, crossed_ma25=True)
    ts = int(datetime(2026, 10, 2, 22, 49, tzinfo=timezone(timedelta(hours=8))).timestamp() * 1000)
    row = ScanRow(
        symbol="ORCLUSDT",
        ts_ms=ts,
        hit=hit,
        fwd15=-1.2,
        fwd30=-2.4,
        low15=-1.5,
        low30=-2.8,
        quote_vol=1.0,
    )
    start, end = day_bounds_ms("2026-10-02")
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "index.html"
        out = write_html_report(
            path, [row], start_ms=start, end_ms=end, n_symbols=1, title="test", top=10
        )
        html = out.read_text(encoding="utf-8")
        assert "ORCLUSDT" in html
        assert "22:49" in html
        payload = json.loads((out.parent / "hits.json").read_text())
        assert payload["count"] == 1
        assert payload["top"][0]["symbol"] == "ORCLUSDT"
        assert payload["orcl_like"] == 1


def test_orcl_like_needs_same_bar_and_lead() -> None:
    ts = 1
    a = ScanRow(
        "A", ts, ShortHit(1, 1, 1, 1, 1, 22, True), -1, -2, -1.5, -2.5, 0
    )
    b = ScanRow(
        "B", ts, ShortHit(1, 1, 1, 1, 1, 22, False), -3, -4, -3, -5, 0
    )
    c = ScanRow(
        "C", ts, ShortHit(1, 1, 1, 1, 1, 3, True), -4, -6, -4, -7, 0
    )
    got = orcl_like([a, b, c])
    assert [r.symbol for r in got] == ["A"]
    shallow = ScanRow(
        "D", ts, ShortHit(1, 1, 1, 1, 1, 22, True), -0.2, -0.1, -0.2, -0.3, 0
    )
    assert orcl_like([a, shallow]) == [a]


def test_new_high_mask() -> None:
    high = np.array([1.0, 2.0, 3.0, 2.0, 4.0, 3.0])
    mask = new_high_mask(high, lookback=2)
    assert list(mask) == [False, False, True, False, True, False]
    assert latest_high_in_window(mask, 5, 1) == 4
    assert latest_high_in_window(mask, 5, 0) is None
    assert latest_high_in_window(mask, 4, 0) == 4
    assert latest_high_in_window(mask, 3, 0) is None
    assert latest_high_in_window(mask, 3, 1) == 2


def test_4h_high_then_death_within_30() -> None:
    n = 300
    close = np.linspace(100.0, 120.0, n - 12)
    dump = np.array([119.5, 118.8, 117.2, 115.4, 113.6, 112.0, 110.6, 109.4, 108.4, 107.6, 107.0, 106.5])
    close = np.concatenate([close, dump])
    high = close + 0.25
    peak = n - 13
    high[peak] = float(close[peak] + 1.8)
    m7, m14, m25 = _mas(close)
    hits = detect_shorts(
        close,
        m7,
        m14,
        m25,
        high=high,
        min_lead=3,
        require_4h_high=True,
        high_lookback=240,
        within_bars=30,
    )
    assert hits, "4h 新高後 30 分內急殺應抓得到"
    hit = hits[-1]
    assert hit.bars_after_high is not None and hit.bars_after_high <= 30
    assert hit.peak_i == peak or high[hit.peak_i] >= high[peak] - 1e-9


def test_4h_high_too_old_skipped() -> None:
    rise = np.linspace(100.0, 120.0, 250)
    wait = np.linspace(120.0, 120.4, 45)
    dump = np.array([120.1, 119.2, 117.6, 115.5, 113.2, 111.0, 109.2, 107.8, 106.6, 105.8, 105.2, 104.8])
    close = np.concatenate([rise, wait, dump])
    high = close + 0.15
    high[249] = 130.0
    m7, m14, m25 = _mas(close)
    raw = detect_shorts(close, m7, m14, m25, min_lead=0, require_4h_high=False)
    gated = detect_shorts(
        close,
        m7,
        m14,
        m25,
        high=high,
        min_lead=0,
        require_4h_high=True,
        high_lookback=240,
        within_bars=30,
    )
    assert raw, "沒加 4h 條件時急殺仍應有死亡交叉"
    late = [h for h in raw if h.i >= 249 + 40]
    assert late, "死亡交叉發生在新高 40 根之後"
    for h in gated:
        assert h.bars_after_high is not None and h.bars_after_high <= 30
    assert all(h.i <= 249 + 30 for h in gated)


def main() -> int:
    test_sma()
    test_death_and_below_fires()
    test_death_above_ma25_skipped()
    test_only_the_cross_bar()
    test_min_lead_filters_flicker()
    test_require_cross_ma25()
    test_lead_bars_counts_pre_cross()
    test_dump_like_screenshot()
    test_filter_universe_keeps_orcl_drops_index()
    test_day_window()
    test_forward_moves_dump()
    test_write_html_report()
    test_orcl_like_needs_same_bar_and_lead()
    test_new_high_mask()
    test_4h_high_then_death_within_30()
    test_4h_high_too_old_skipped()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

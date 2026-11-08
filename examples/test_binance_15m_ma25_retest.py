#!/usr/bin/env python3
"""Synthetic tests for 幣安 15 分站上 MA25 進場（不打幣安）。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from binance_15m_ma25_retest import (  # noqa: E402
    Params,
    Trade,
    _mas,
    chart_payload,
    detect_trades,
    manage_exit,
    summarize,
)


def _ohlc(closes: list[float], wick: float = 0.05) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    c = np.array(closes, dtype=float)
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) + wick
    l = np.minimum(o, c) - wick
    return o, h, l, c


def _picture_setup() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """破底後拉回，第一根陽線同時站上六條且 MA7 > MA14 > MA25。"""
    closes = [100.0] * 210 + [88, 94, 98, 103, 108, 112, 116, 120, 124, 128]
    o, h, l, c = _ohlc(closes)
    entry_i = 215
    l[entry_i] = 110.0
    h[entry_i] = 112.4
    for j in range(entry_i + 1, len(c)):
        l[j] = max(l[j], 110.5)
    h[entry_i + 1] = max(h[entry_i + 1], 116.4)
    return o, h, l, c, entry_i


def test_picture_reclaim_is_the_entry() -> None:
    o, h, l, c, entry_i = _picture_setup()
    mas = _mas(c)
    assert c[entry_i] > o[entry_i]
    assert all(c[entry_i] > mas[n][entry_i] for n in mas)
    assert mas[7][entry_i] > mas[14][entry_i] > mas[25][entry_i]
    assert all(c[213] > mas[n][213] for n in mas)
    assert not (mas[7][213] > mas[14][213] > mas[25][213])
    risk = (c[entry_i] - l[entry_i]) / c[entry_i]
    assert 0.002 <= risk <= 0.04

    funnel: dict = {}
    trades = detect_trades(o, h, l, c, Params(), funnel=funnel)
    assert trades, funnel
    trade = trades[0]
    assert trade.entry_i == entry_i
    assert trade.reclaim_i == entry_i
    assert trade.entry_i > trade.trough_i
    assert trade.reason == "2R"
    assert trade.pnl_pct > 0
    assert funnel["trades"] == 1
    assert funnel["reclaim"] == 1


def test_ma25_cross_under_long_ma_is_not_entry() -> None:
    closes = [100.0] * 210 + [90.0] + [92.0] * 20 + [96.0, 97.0, 99.0]
    o, h, l, c = _ohlc(closes)
    mas = _mas(c)
    poke = 231
    assert c[poke] > mas[25][poke]
    assert c[poke] < mas[200][poke]
    trades = detect_trades(o, h, l, c, Params())
    assert trades == []


def test_stays_under_ma25_has_no_trade() -> None:
    closes = [100.0] * 210 + [90.0] + [92.0] * 20
    o, h, l, c = _ohlc(closes)
    trades = detect_trades(o, h, l, c, Params())
    assert trades == []


def test_new_low_on_the_cross_bar_is_not_entry() -> None:
    closes = [100.0] * 210 + [90.0, 96.0]
    o, h, l, c = _ohlc(closes)
    cross = 211
    l[cross] = l[210] - 1.0
    c[cross] = 96.0
    trades = detect_trades(o, h, l, c, Params())
    assert trades == []


def test_wide_stop_is_skipped() -> None:
    o, h, l, c, entry_i = _picture_setup()
    l[entry_i] = c[entry_i] * 0.90
    trades = detect_trades(o, h, l, c, Params())
    assert all(t.entry_i != entry_i for t in trades)


def test_stop_before_target_on_same_bar() -> None:
    o = np.array([10.0, 10.2, 9.0])
    h = np.array([10.3, 12.0, 11.0])
    l = np.array([9.8, 8.5, 8.8])
    c = np.array([10.1, 11.0, 9.2])
    exit_i, px, reason = manage_exit(o, h, l, c, 0, 10.0, 9.5, 12.0, time_stop_bars=5)
    assert exit_i == 1
    assert reason == "停損"
    assert px == 9.5


def test_summarize_splits_open_trades() -> None:
    closed = Trade("", 1, 2, 3, 4, 90, 100, 99, 102, 102, "2R", 2.0, 1.0)
    opened = Trade("", 1, 2, 5, 6, 90, 100, 99, 102, 99.5, "未平倉", -0.5, 1.0)
    stats = summarize([closed, opened])
    assert stats["count"] == 1
    assert stats["open"] == 1
    assert stats["wins"] == 1
    assert stats["win_rate"] == 100.0
    assert abs(stats["sum_pct"] - 2.0) < 1e-9
    assert abs(stats["sum_all_pct"] - 1.5) < 1e-9


def test_chart_payload_aligns_marks() -> None:
    o, h, l, c, entry_i = _picture_setup()
    trades = detect_trades(o, h, l, c, Params())
    assert trades
    trade = trades[0]
    trade.symbol = "TESTUSDT"
    times = [1_700_000_000_000 + i * 900_000 for i in range(len(c))]
    bars = {"o": o, "h": h, "l": l, "c": c}
    payload = chart_payload(bars, trade, times, 1)
    n = len(payload["c"])
    assert n == len(payload["o"]) == len(payload["m25"]) == len(payload["m200"])
    trough_x, entry_x, exit_x = payload["marks"]
    assert 0 <= trough_x < entry_x < n
    assert 0 <= exit_x < n
    assert payload["c"][entry_x] > payload["m25"][entry_x]
    assert payload["c"][entry_x] > payload["m200"][entry_x]
    assert n > entry_x + 3
    assert payload["symbol"] == "TESTUSDT"


def test_min_entry_index_filters_warmup() -> None:
    o, h, l, c, entry_i = _picture_setup()
    early = detect_trades(o, h, l, c, Params(), min_entry_i=0)
    assert early
    assert early[0].entry_i == entry_i
    late = detect_trades(o, h, l, c, Params(), min_entry_i=entry_i + 1)
    assert late == []


def main() -> int:
    test_picture_reclaim_is_the_entry()
    test_ma25_cross_under_long_ma_is_not_entry()
    test_stays_under_ma25_has_no_trade()
    test_new_low_on_the_cross_bar_is_not_entry()
    test_wide_stop_is_skipped()
    test_stop_before_target_on_same_bar()
    test_summarize_splits_open_trades()
    test_chart_payload_aligns_marks()
    test_min_entry_index_filters_warmup()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

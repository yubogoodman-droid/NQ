#!/usr/bin/env python3
"""Synthetic tests for 幣安 15 分六條均線回測（不打幣安）。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from binance_15m_ma25_retest import (  # noqa: E402
    Params,
    Trade,
    _mas,
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


def _base_closes() -> list[float]:
    """200 根平盤讓六條均線就緒，再破底、拉回全部均線之上。"""
    closes = [100.0] * 210
    closes += [96, 92, 88, 90, 96, 103, 108, 112, 116]
    return closes


def _arm_retest(closes: list[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """回傳 ohlc、站回索引、預備回測的那根索引。回測低點留給呼叫端改。"""
    o, h, l, c = _ohlc(closes)
    mas = _mas(c)
    reclaim = next(
        i
        for i in range(210, len(c))
        if all(c[i] > mas[n][i] for n in mas) and mas[7][i] > mas[14][i] > mas[25][i]
    )
    ext = reclaim + 1
    assert ext < len(c)
    l[ext] = float(mas[7][ext]) + 0.2
    h[ext] = max(h[ext], c[ext] + 0.05)
    return o, h, l, c, reclaim, ext


def test_reclaim_all_then_retest_hits_2r() -> None:
    closes = _base_closes() + [115.0, 104.5, 108, 112, 118, 124]
    o, h, l, c, reclaim, ext = _arm_retest(closes)
    mas = _mas(c)
    entry_i = ext + 1
    l[entry_i] = float(mas[25][entry_i]) - 0.08
    assert c[entry_i] > mas[99][entry_i]
    assert c[entry_i] > mas[200][entry_i]
    assert mas[7][entry_i] > mas[14][entry_i] > mas[25][entry_i]
    risk = (c[entry_i] - l[entry_i]) / c[entry_i]
    assert 0.002 <= risk <= 0.04

    funnel: dict = {}
    trades = detect_trades(o, h, l, c, Params(), funnel=funnel)
    assert trades, funnel
    trade = trades[0]
    assert trade.entry_i == entry_i
    assert trade.reclaim_i == reclaim
    assert trade.entry_i > trade.reclaim_i > trade.trough_i
    assert trade.reason == "2R"
    assert trade.pnl_pct > 0
    assert funnel["trades"] == 1


def test_close_back_under_long_ma_cancels() -> None:
    closes = _base_closes() + [115.0, 90.0, 92.0, 94.0]
    o, h, l, c, _reclaim, ext = _arm_retest(closes)
    mas = _mas(c)
    lose = ext + 1
    c[lose] = float(mas[25][lose]) - 1.0
    o[lose] = float(mas[25][lose]) + 0.2
    h[lose] = o[lose] + 0.1
    l[lose] = c[lose] - 0.1
    trades = detect_trades(o, h, l, c, Params())
    assert trades == []


def test_touch_before_leaving_ma7_is_not_entry() -> None:
    closes = _base_closes() + [114.0]
    o, h, l, c, reclaim, ext = _arm_retest(closes)
    mas = _mas(c)
    # 站回後下一根立刻刺到 MA25，還沒有低點站上 MA7
    l[ext] = float(mas[25][ext]) - 0.2
    c[ext] = float(max(mas[200][ext], mas[25][ext])) + 1.5
    o[ext] = c[ext]
    h[ext] = c[ext] + 0.2
    trades = detect_trades(o, h, l, c, Params())
    assert all(t.entry_i != ext for t in trades)
    assert all(t.reclaim_i != reclaim or t.entry_i > ext for t in trades)


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


def test_min_entry_index_filters_warmup() -> None:
    closes = _base_closes() + [115.0, 104.5, 108, 112, 118, 124]
    o, h, l, c, _reclaim, ext = _arm_retest(closes)
    mas = _mas(c)
    entry_i = ext + 1
    l[entry_i] = float(mas[25][entry_i]) - 0.08
    early = detect_trades(o, h, l, c, Params(), min_entry_i=0)
    assert early
    late = detect_trades(o, h, l, c, Params(), min_entry_i=early[0].entry_i + 1)
    assert late == []


def main() -> int:
    test_reclaim_all_then_retest_hits_2r()
    test_close_back_under_long_ma_cancels()
    test_touch_before_leaving_ma7_is_not_entry()
    test_stop_before_target_on_same_bar()
    test_summarize_splits_open_trades()
    test_min_entry_index_filters_warmup()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

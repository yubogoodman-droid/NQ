#!/usr/bin/env python3
"""Synthetic tests for 幣安 15 分破底站回 MA25 回測（不打幣安）。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from binance_15m_ma25_retest import (  # noqa: E402
    Params,
    detect_trades,
    manage_exit,
    summarize,
)


def _ohlc(closes: list[float], wick: float = 0.15) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    c = np.array(closes, dtype=float)
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) + wick
    l = np.minimum(o, c) - wick
    return o, h, l, c


def _flat_then_break() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """40 根平盤後破底，站回，站穩，再回測。回測之後拉高到足以打 2R。"""
    closes = [100.0] * 40
    closes += [97, 94, 90, 93, 97, 101.5]  # 破底到 90，再站回
    closes += [102.2]  # 站穩，低點會在收盤附近、高於當時 MA
    closes += [101.2]  # 回測候選，低點等下改成刺到 MA
    closes += [103, 105, 108, 112, 116, 120]
    o, h, l, c = _ohlc(closes, wick=0.05)
    # 站穩那根的低點抬到收盤附近，避免刺到均線
    hold = 46
    l[hold] = min(o[hold], c[hold]) + 0.02
    h[hold] = max(o[hold], c[hold]) + 0.05
    return o, h, l, c


def test_reclaim_then_retest_hits_2r() -> None:
    o, h, l, c = _flat_then_break()
    from binance_15m_ma25_retest import sma

    ma = sma(c, 25)
    entry_guess = 47
    assert c[entry_guess] > ma[entry_guess]
    l[entry_guess] = float(ma[entry_guess]) - 0.05
    # 風險要落在 0.2%–4%
    assert 0.002 <= (c[entry_guess] - l[entry_guess]) / c[entry_guess] <= 0.04

    funnel: dict = {}
    trades = detect_trades(o, h, l, c, Params(), funnel=funnel)
    assert trades, funnel
    trade = trades[0]
    assert trade.entry_i > trade.reclaim_i > trade.trough_i
    assert trade.reason == "2R"
    assert trade.pnl_pct > 0
    assert trade.entry == c[trade.entry_i]
    assert trade.stop == l[trade.entry_i]
    assert funnel["trades"] == 1


def test_close_back_under_ma_cancels() -> None:
    closes = [100.0] * 40 + [96, 92, 88, 94, 101.5, 102.0, 99.0, 98.0, 97.0]
    o, h, l, c = _ohlc(closes, wick=0.05)
    from binance_15m_ma25_retest import sma

    ma = sma(c, 25)
    # 站回後下一根站穩，再下一根收盤跌回均線下
    hold = None
    for i in range(45, len(c)):
        if c[i] > ma[i] and l[i] > ma[i]:
            hold = i
            break
    assert hold is not None
    lose = hold + 1
    c[lose] = float(ma[lose]) - 0.4
    o[lose] = float(ma[lose]) + 0.1
    h[lose] = o[lose] + 0.05
    l[lose] = c[lose] - 0.05
    trades = detect_trades(o, h, l, c, Params())
    assert trades == []


def test_touch_before_standing_is_not_entry() -> None:
    """站回後立刻刺回均線、還沒有一根完全站穩，不當進場。"""
    closes = [100.0] * 40 + [96, 92, 88, 94, 102.0]
    o, h, l, c = _ohlc(closes, wick=0.05)
    from binance_15m_ma25_retest import sma

    ma = sma(c, 25)
    reclaim = next(i for i in range(44, len(c)) if c[i] > ma[i])
    # 下一根立刻刺到均線又收上，沒有先站穩
    k = reclaim + 1
    c = np.r_[c, c[-1] + 0.2]
    o = np.r_[o, c[-2]]
    h = np.r_[h, c[-1] + 0.05]
    l = np.r_[l, float(ma[reclaim]) - 0.2]
    # 對齊新的一根：低點刺到當根均線
    ma2 = sma(c, 25)
    l[-1] = float(ma2[-1]) - 0.05
    c[-1] = float(ma2[-1]) + 0.3
    o[-1] = c[-1]
    h[-1] = c[-1] + 0.05
    trades = detect_trades(o, h, l, c, Params())
    assert all(t.entry_i != k for t in trades)


def test_stop_before_target_on_same_bar() -> None:
    o = np.array([10.0, 10.2, 9.0])
    h = np.array([10.3, 12.0, 11.0])
    l = np.array([9.8, 8.5, 8.8])
    c = np.array([10.1, 11.0, 9.2])
    # 進場在 index 0 之後的下一根同時打到停損 9.5 和目標 12
    exit_i, px, reason = manage_exit(o, h, l, c, 0, 10.0, 9.5, 12.0, time_stop_bars=5)
    assert exit_i == 1
    assert reason == "停損"
    assert px == 9.5


def test_summarize_splits_open_trades() -> None:
    from binance_15m_ma25_retest import Trade

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
    o, h, l, c = _flat_then_break()
    from binance_15m_ma25_retest import sma

    ma = sma(c, 25)
    entry_guess = 47
    l[entry_guess] = float(ma[entry_guess]) - 0.05
    early = detect_trades(o, h, l, c, Params(), min_entry_i=0)
    assert early
    late = detect_trades(o, h, l, c, Params(), min_entry_i=early[0].entry_i + 1)
    assert late == []


def main() -> int:
    test_reclaim_then_retest_hits_2r()
    test_close_back_under_ma_cancels()
    test_touch_before_standing_is_not_entry()
    test_stop_before_target_on_same_bar()
    test_summarize_splits_open_trades()
    test_min_entry_index_filters_warmup()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

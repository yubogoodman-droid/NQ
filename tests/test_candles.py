"""一分 K 棒型態與回測單元測試。"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from nq.candles import detect_candle_patterns
from nq.one_min import OneMinCandleStrategy, run_one_min_backtest


def _frame(rows: list[tuple[float, float, float, float, float]]) -> pd.DataFrame:
    idx = pd.date_range("2026-08-21 10:00", periods=len(rows), freq="1min")
    data = np.array(rows, dtype=float)
    return pd.DataFrame(
        {"open": data[:, 0], "high": data[:, 1], "low": data[:, 2], "close": data[:, 3], "volume": data[:, 4]},
        index=idx,
    )


def _pad(n: int = 40, price: float = 100.0) -> list[tuple[float, float, float, float, float]]:
    rows = []
    px = price
    for i in range(n):
        o = px
        c = px + 0.02
        h = c + 0.03
        l = o - 0.03 - i * 0.0002
        rows.append((o, h, l, c, 500))
        px = c
    return rows


class CandlePatternTests(unittest.TestCase):
    def test_hammer_detected(self) -> None:
        rows = _pad(30, 100.0)
        rows.append((99.6, 99.7, 98.4, 99.55, 800))
        df = _frame(rows)
        names = [p.name for p in detect_candle_patterns(df, include_range_breakout=False)]
        self.assertIn("hammer", names)

    def test_bull_engulfing_detected(self) -> None:
        rows = _pad(30, 100.0)
        rows.append((100.4, 100.5, 99.6, 99.7, 600))
        rows.append((99.6, 100.7, 99.5, 100.6, 900))
        df = _frame(rows)
        names = [p.name for p in detect_candle_patterns(df, include_range_breakout=False)]
        self.assertIn("bull_engulfing", names)

    def test_range_breakout_detected(self) -> None:
        rows = []
        for i in range(70):
            base = 100.0 + 0.45 * np.sin(i / 5)
            rows.append((base, base + 0.30, base - 0.30, base + 0.04, 400))
        rows.append((100.3, 102.4, 100.2, 102.2, 2000))
        df = _frame(rows)
        names = [p.name for p in detect_candle_patterns(df)]
        self.assertIn("range_breakout", names)

    def test_chase_filter_skips_extended_move(self) -> None:
        rows = []
        px = 100.0
        for _ in range(40):
            px *= 1.003
            rows.append((px * 0.999, px * 1.002, px * 0.998, px, 800))
        rows.append((px, px + 0.1, px - 1.4, px + 0.05, 900))
        df = _frame(rows)
        patterns = detect_candle_patterns(df, max_chase_pct=0.06, include_range_breakout=False)
        self.assertFalse(any(p.name == "hammer" for p in patterns))


class OneMinBacktestTests(unittest.TestCase):
    def test_next_bar_entry_and_take_profit(self) -> None:
        rows = _pad(30, 100.0)
        rows.append((99.6, 99.7, 98.4, 99.55, 800))  # hammer
        rows.append((99.6, 99.65, 99.5, 99.62, 500))  # 進場開盤
        # 後面抬高打到目標
        for _ in range(8):
            rows.append((99.8, 103.5, 99.7, 103.0, 700))
        df = _frame(rows)
        strategy = OneMinCandleStrategy(tick_size=0.01, reward_r=1.5, max_bars_hold=12)
        trades = run_one_min_backtest(df, symbol="TEST", strategy=strategy, cost_bps=0)
        hammer_trades = [t for t in trades if t.signal.pattern.name == "hammer"]
        self.assertTrue(hammer_trades)
        trade = hammer_trades[0]
        self.assertEqual(trade.signal.bar_idx, 31)
        self.assertAlmostEqual(trade.signal.entry, 99.60, places=2)
        self.assertIn(trade.exit_reason, {"take_profit", "time_stop"})

    def test_stop_checked_before_target(self) -> None:
        rows = _pad(30, 100.0)
        rows.append((99.6, 99.7, 98.4, 99.55, 800))
        # 進場當根同時打到停損與更高價，應記停損
        rows.append((99.6, 110.0, 90.0, 99.0, 900))
        df = _frame(rows)
        strategy = OneMinCandleStrategy(tick_size=0.01)
        trades = run_one_min_backtest(df, symbol="TEST", strategy=strategy, cost_bps=0)
        hammer_trades = [t for t in trades if t.signal.pattern.name == "hammer"]
        self.assertTrue(hammer_trades)
        self.assertEqual(hammer_trades[0].exit_reason, "stop_loss")
        self.assertLess(hammer_trades[0].pnl_points, 0)


if __name__ == "__main__":
    unittest.main()

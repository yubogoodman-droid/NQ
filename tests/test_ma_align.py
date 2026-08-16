"""日線 MA5/10/20 多頭排列且站上 MA200 的單元測試。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nq.ma_align import add_daily_mas, detect_ma_align_alerts, is_aligned
from nq.strategy import MaAlignStrategy


def _daily(closes: list[float], start: str = "2025-01-02") -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=len(closes))
    rows = [(c * 0.999, c * 1.002, c * 0.998, c, 1_000_000) for c in closes]
    return pd.DataFrame(rows, index=idx, columns=["open", "high", "low", "close", "volume"])


class MaAlignTests(unittest.TestCase):
    def test_fires_when_stack_and_close_crosses_ma200(self) -> None:
        closes = [100.0] * 200
        for i in range(1, 25):
            closes.append(100.0 + i * 0.8)
        df = _daily(closes)
        alerts = detect_ma_align_alerts(df)
        self.assertGreaterEqual(len(alerts), 1)
        p = alerts[0]
        self.assertGreater(p.ma5, p.ma10)
        self.assertGreater(p.ma10, p.ma20)
        self.assertGreater(p.close, p.ma200)
        row = add_daily_mas(df).iloc[p.bar_idx]
        self.assertTrue(is_aligned(row))
        if p.bar_idx > 0:
            self.assertFalse(is_aligned(add_daily_mas(df).iloc[p.bar_idx - 1]))

        signals = MaAlignStrategy(tick_size=0.01).generate_signals(df)
        self.assertGreaterEqual(len(signals), 1)
        sig = signals[0]
        self.assertGreater(sig.entry, sig.stop_loss)
        self.assertGreater(sig.target, sig.entry)

    def test_does_not_repeat_while_still_aligned(self) -> None:
        closes = [100.0] * 200 + [100.0 + i * 0.8 for i in range(1, 40)]
        df = _daily(closes)
        alerts = detect_ma_align_alerts(df)
        self.assertGreaterEqual(len(alerts), 1)
        idxs = [p.bar_idx for p in alerts]
        for a, b in zip(idxs, idxs[1:]):
            self.assertGreater(b - a, 1)

    def test_ignores_stack_below_ma200(self) -> None:
        closes = [150.0] * 200 + [90.0 + i * 0.4 for i in range(40)]
        df = _daily(closes)
        work = add_daily_mas(df)
        last = work.iloc[-1]
        self.assertGreater(float(last["ma5"]), float(last["ma10"]))
        self.assertGreater(float(last["ma10"]), float(last["ma20"]))
        self.assertLess(float(last["close"]), float(last["ma200"]))
        self.assertEqual(detect_ma_align_alerts(df), [])


if __name__ == "__main__":
    unittest.main()

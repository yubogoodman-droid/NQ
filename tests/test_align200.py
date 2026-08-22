"""5>10>20>60 站上 MA200 訊號測試。"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from nq.align200 import detect_align200, run_align200_backtest


def _frame(closes: list[float]) -> pd.DataFrame:
    n = len(closes)
    idx = pd.date_range("2026-08-21 09:00", periods=n, freq="1min", tz="Asia/Taipei")
    c = np.array(closes, dtype=float)
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame(
        {
            "open": o,
            "high": np.maximum(o, c) + 0.1,
            "low": np.minimum(o, c) - 0.1,
            "close": c,
            "volume": np.full(n, 500.0),
        },
        index=idx,
    )


class Align200Tests(unittest.TestCase):
    def test_fires_when_alignment_and_cross_200(self) -> None:
        # 先在 100 附近盤，讓 MA200≈100，再拉到 102 並讓短均排好
        closes = [100.0 + 0.02 * np.sin(i / 8) for i in range(220)]
        last = closes[-1]
        for _ in range(25):
            last += 0.18
            closes.append(last)
        df = _frame(closes)
        sigs = detect_align200(df, symbol="2408.TW", name="南亞科")
        self.assertTrue(sigs)
        s = sigs[0]
        self.assertGreater(s.ma5, s.ma10)
        self.assertGreater(s.ma10, s.ma20)
        self.assertGreater(s.ma20, s.ma60)
        self.assertGreater(s.close, s.ma200)

    def test_does_not_repeat_same_day(self) -> None:
        closes = [100.0 + 0.02 * np.sin(i / 8) for i in range(220)]
        last = closes[-1]
        for _ in range(40):
            last += 0.15
            closes.append(last)
        df = _frame(closes)
        sigs = detect_align200(df, symbol="2408.TW")
        self.assertEqual(len(sigs), 1)

    def test_backtest_has_exit(self) -> None:
        closes = [100.0 + 0.02 * np.sin(i / 8) for i in range(220)]
        last = closes[-1]
        for _ in range(20):
            last += 0.2
            closes.append(last)
        for _ in range(15):
            last -= 0.35
            closes.append(last)
        df = _frame(closes)
        sigs = detect_align200(df, symbol="2408.TW")
        trades = run_align200_backtest(df, sigs, cost_bps=0)
        self.assertTrue(trades)
        self.assertIn(trades[0].exit_reason, {"lost_align", "session_flat", "time_stop"})


if __name__ == "__main__":
    unittest.main()

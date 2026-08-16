"""1 分 K：MA5/10/20 多頭排列且收盤站上 MA200。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nq.ma_align import add_mas, detect_ma_align_alerts, is_aligned
from nq.strategy import MaAlignStrategy


def _bars(closes: list[float], start: str = "2026-08-14 09:00") -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(closes), freq="1min")
    rows = [(c * 0.999, c * 1.002, c * 0.998, c, 1_000_000) for c in closes]
    return pd.DataFrame(rows, index=idx, columns=["open", "high", "low", "close", "volume"])


class MaAlignTests(unittest.TestCase):
    def test_fires_when_stack_and_close_crosses_ma200(self) -> None:
        closes = [100.0] * 200
        for i in range(1, 25):
            closes.append(100.0 + i * 0.8)
        df = _bars(closes, start="2026-08-14 10:00")
        alerts = detect_ma_align_alerts(df, skip_open_minutes=0)
        self.assertGreaterEqual(len(alerts), 1)
        p = alerts[0]
        self.assertGreater(p.ma5, p.ma10)
        self.assertGreater(p.ma10, p.ma20)
        self.assertGreater(p.close, p.ma200)
        row = add_mas(df).iloc[p.bar_idx]
        self.assertTrue(is_aligned(row))
        if p.bar_idx > 0:
            prev = add_mas(df).iloc[p.bar_idx - 1]
            self.assertFalse(float(prev["close"]) > float(prev["ma200"]))

        signals = MaAlignStrategy(tick_size=0.01, skip_open_minutes=0).generate_signals(df)
        self.assertGreaterEqual(len(signals), 1)
        sig = signals[0]
        self.assertGreater(sig.entry, sig.stop_loss)
        self.assertGreater(sig.target, sig.entry)

    def test_does_not_repeat_while_still_aligned(self) -> None:
        closes = [100.0] * 200 + [100.0 + i * 0.8 for i in range(1, 40)]
        df = _bars(closes, start="2026-08-14 10:00")
        alerts = detect_ma_align_alerts(df, skip_open_minutes=0)
        self.assertGreaterEqual(len(alerts), 1)
        idxs = [p.bar_idx for p in alerts]
        for a, b in zip(idxs, idxs[1:]):
            self.assertGreater(b - a, 1)

    def test_ignores_stack_below_ma200(self) -> None:
        closes = [150.0] * 200 + [90.0 + i * 0.4 for i in range(40)]
        df = _bars(closes, start="2026-08-14 10:00")
        work = add_mas(df)
        last = work.iloc[-1]
        self.assertGreater(float(last["ma5"]), float(last["ma10"]))
        self.assertGreater(float(last["ma10"]), float(last["ma20"]))
        self.assertLess(float(last["close"]), float(last["ma200"]))
        self.assertEqual(detect_ma_align_alerts(df, skip_open_minutes=0), [])

    def test_ignores_restack_while_already_above_ma200(self) -> None:
        closes = [100.0] * 200 + [130.0] * 12 + [115.0] * 10 + [130.0 + i * 0.2 for i in range(20)]
        df = _bars(closes, start="2026-08-14 10:00")
        work = add_mas(df)
        alerts = detect_ma_align_alerts(df, skip_open_minutes=0)
        self.assertGreaterEqual(len(alerts), 1)
        first = alerts[0]
        self.assertGreater(float(work.iloc[first.bar_idx]["close"]), float(work.iloc[first.bar_idx]["ma200"]))
        restack_start = 200 + 12 + 10
        later = [p for p in alerts if p.bar_idx >= restack_start]
        self.assertEqual(later, [])
        last = work.iloc[-1]
        self.assertGreater(float(last["close"]), float(last["ma200"]))
        self.assertGreater(float(last["ma5"]), float(last["ma10"]))
        self.assertGreater(float(last["ma10"]), float(last["ma20"]))

    def test_fires_again_after_close_drops_below_ma200(self) -> None:
        closes = [100.0] * 200 + [120.0] * 8 + [80.0] * 30 + [130.0] * 15
        df = _bars(closes, start="2026-08-14 10:00")
        work = add_mas(df)
        alerts = detect_ma_align_alerts(df, skip_open_minutes=0)
        self.assertGreaterEqual(len(alerts), 2)
        for p in alerts:
            if p.bar_idx > 0:
                prev = work.iloc[p.bar_idx - 1]
                self.assertFalse(float(prev["close"]) > float(prev["ma200"]))
            row = work.iloc[p.bar_idx]
            self.assertGreater(float(row["close"]), float(row["ma200"]))
            self.assertGreater(float(row["ma5"]), float(row["ma10"]))
            self.assertGreater(float(row["ma10"]), float(row["ma20"]))

    def test_skips_first_minutes(self) -> None:
        closes = [100.0] * 200 + [100.0 + i * 0.8 for i in range(1, 20)]
        df = _bars(closes, start="2026-08-14 09:00")
        early = detect_ma_align_alerts(df, skip_open_minutes=5)
        for p in early:
            t = df.index[p.bar_idx]
            self.assertGreaterEqual(t.hour * 60 + t.minute, 9 * 60 + 5)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

import pandas as pd

from tw.signals import add_moving_averages, is_ma200_breakout_bullish


def _bars(closes: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2026-08-17 09:00", periods=len(closes), freq="1min", tz="Asia/Taipei")
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c + 0.5 for c in closes],
            "low": [c - 0.5 for c in closes],
            "close": closes,
            "volume": [1000] * len(closes),
        },
        index=idx,
    )


class SignalTests(unittest.TestCase):
    def test_breakout_with_bullish_ma_alignment(self) -> None:
        closes = [100.0] * 200 + [110.0]
        snap = is_ma200_breakout_bullish(_bars(closes))
        self.assertIsNotNone(snap)
        assert snap is not None
        self.assertGreater(snap.ma5, snap.ma10)
        self.assertGreater(snap.ma10, snap.ma20)
        self.assertGreater(snap.close, snap.ma200)
        self.assertLessEqual(snap.prev_close, snap.prev_ma200)

    def test_already_above_previous_bar_is_not_new_signal(self) -> None:
        closes = [100.0] * 198
        closes.extend([104.0, 105.0, 106.0, 107.0])
        snap = is_ma200_breakout_bullish(_bars(closes))
        self.assertIsNone(snap)

    def test_no_signal_without_bullish_alignment(self) -> None:
        # 最後一根剛站上 MA200，但短均不是多頭排列
        closes = [100.0] * 197 + [80.0, 80.0, 80.0, 110.0]
        snap = is_ma200_breakout_bullish(_bars(closes))
        self.assertIsNone(snap)

    def test_need_201_bars(self) -> None:
        self.assertIsNone(is_ma200_breakout_bullish(_bars([100.0] * 200)))

    def test_moving_averages_length(self) -> None:
        df = add_moving_averages(_bars([float(i) for i in range(1, 221)]))
        self.assertTrue(pd.isna(df["ma200"].iloc[198]))
        self.assertFalse(pd.isna(df["ma200"].iloc[199]))


if __name__ == "__main__":
    unittest.main()

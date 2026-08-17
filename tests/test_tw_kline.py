from __future__ import annotations

import unittest

import pandas as pd

from tw.kline import _normalize_ohlcv, _slice_ticker


class KlineNormalizeTests(unittest.TestCase):
    def test_normalize_yahoo_columns(self) -> None:
        idx = pd.date_range("2026-08-17 09:00", periods=3, freq="1min", tz="Asia/Taipei")
        df = pd.DataFrame(
            {
                "Open": [10.0, 11.0, 12.0],
                "High": [10.5, 11.5, 12.5],
                "Low": [9.5, 10.5, 11.5],
                "Close": [10.2, 11.1, 12.2],
                "Volume": [1, 2, 3],
            },
            index=idx,
        )
        out = _normalize_ohlcv(df)
        self.assertEqual(list(out.columns), ["open", "high", "low", "close", "volume"])
        self.assertEqual(len(out), 3)
        self.assertEqual(out["close"].iloc[-1], 12.2)

    def test_slice_multiindex_ticker(self) -> None:
        idx = pd.date_range("2026-08-17 09:00", periods=2, freq="1min", tz="Asia/Taipei")
        cols = pd.MultiIndex.from_product([["2408.TW", "1303.TW"], ["Open", "Close"]])
        raw = pd.DataFrame(
            [[1, 2, 3, 4], [5, 6, 7, 8]],
            index=idx,
            columns=cols,
        )
        sub = _slice_ticker(raw, "1303.TW")
        self.assertIn("Close", sub.columns)
        self.assertEqual(sub["Close"].iloc[-1], 8)

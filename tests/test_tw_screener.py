from __future__ import annotations

import unittest
from zoneinfo import ZoneInfo

import pandas as pd

from tw.ranking import RankedStock
from tw.screener import ScanHit, apply_5m_ma200_filter
from tw.signals import AlertSnapshot


TAIPEI = ZoneInfo("Asia/Taipei")


def _bars(closes: list[float], freq: str = "5min") -> pd.DataFrame:
    idx = pd.date_range("2026-08-17 09:00", periods=len(closes), freq=freq, tz=TAIPEI)
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c + 0.5 for c in closes],
            "low": [c - 0.5 for c in closes],
            "close": closes,
            "volume": [1000.0] * len(closes),
        },
        index=idx,
    )


def _hit(symbol: str, frame_5m: pd.DataFrame | None, ts: pd.Timestamp) -> ScanHit:
    return ScanHit(
        stock=RankedStock(1, symbol, symbol, 100.0, 0.0, 0.0, 1, 1e9, "TAI"),
        snapshot=AlertSnapshot(
            timestamp=ts,
            close=110.0,
            prev_close=100.0,
            ma5=102.0,
            ma10=101.0,
            ma20=100.5,
            ma200=100.05,
            prev_ma200=100.0,
        ),
        bars=240,
        frame=None,
        frame_5m=frame_5m,
    )


class FiveMinuteFilterTests(unittest.TestCase):
    def test_drops_hits_below_5m_ma200(self) -> None:
        above_df = _bars([100.0] * 219 + [110.0])
        below_df = _bars([100.0] * 219 + [90.0])
        ts = above_df.index[-1] + pd.Timedelta(minutes=3)
        above = _hit("1303.TW", above_df, ts)
        below = _hit("2313.TW", below_df, ts)
        missing = _hit("2486.TW", None, ts)
        kept, dropped, skipped = apply_5m_ma200_filter([above, below, missing])
        self.assertEqual([h.stock.symbol for h in kept], ["1303.TW"])
        self.assertEqual(dropped, 2)
        reasons = {stock.symbol: reason for stock, reason in skipped}
        self.assertIn("MA200", reasons["2313.TW"])
        self.assertIn("無五分", reasons["2486.TW"])


if __name__ == "__main__":
    unittest.main()

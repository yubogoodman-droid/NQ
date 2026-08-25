from __future__ import annotations

import unittest
from zoneinfo import ZoneInfo

import pandas as pd

from tw.ranking import RankedStock
from tw.screener import ScanHit, hit_key
from tw.signals import AlertSnapshot, close_above_ma240


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


def _hit(
    symbol: str,
    frame_5m: pd.DataFrame | None,
    ts: pd.Timestamp,
    *,
    close: float = 110.0,
    ma5: float = 109.4,
    ma10: float = 109.1,
    ma20: float = 108.8,
) -> ScanHit:
    return ScanHit(
        stock=RankedStock(1, symbol, symbol, 100.0, 0.0, 0.0, 1, 1e9, "TAI"),
        snapshot=AlertSnapshot(
            timestamp=ts,
            close=close,
            prev_close=close - 1,
            ma5=ma5,
            ma10=ma10,
            ma20=ma20,
            ma240=close - 0.5,
            prev_ma240=close - 0.4,
        ),
        bars=240,
        frame=None,
        frame_5m=frame_5m,
    )


class ChartOnlyFiveMinuteTests(unittest.TestCase):
    def test_5m_below_ma240_does_not_drop_1m_hit(self) -> None:
        below_df = _bars([100.0] * 259 + [90.0])
        ts = below_df.index[-1] + pd.Timedelta(minutes=3)
        hit = _hit("2313.TW", below_df, ts)
        self.assertFalse(close_above_ma240(hit.frame_5m, hit.snapshot.timestamp, floor="5min"))
        self.assertTrue(hit.snapshot.bullish_aligned)
        self.assertTrue(hit.snapshot.crossed_above_ma240)

    def test_missing_5m_does_not_drop_1m_hit(self) -> None:
        ts = pd.Timestamp("2026-08-17 11:03", tz=TAIPEI)
        hit = _hit("2486.TW", None, ts)
        self.assertIsNone(hit.frame_5m)
        self.assertTrue(hit.snapshot.crossed_above_ma240)

    def test_hit_key_uses_symbol_and_timestamp(self) -> None:
        ts = pd.Timestamp("2026-08-17 11:03", tz=TAIPEI)
        hit = _hit("1303.TW", None, ts)
        self.assertEqual(hit_key(hit), ("1303.TW", ts))


if __name__ == "__main__":
    unittest.main()

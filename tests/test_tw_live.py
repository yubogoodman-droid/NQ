from __future__ import annotations

import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from tw.live import (
    BarAggregator,
    OhlcvBar,
    alerts_on_closed_bar,
    floor_bar,
    format_telegram,
    in_session,
    kbars_to_ohlcv,
    should_run_15m,
    upsert_bar,
)
from tw.signals import iter_5m_ma200_alerts
from tests.test_tw_5m import _history_then_live

TAIPEI = ZoneInfo("Asia/Taipei")


class LiveHelperTests(unittest.TestCase):
    def test_floor_bar_to_five_minutes(self) -> None:
        ts = pd.Timestamp("2026-08-21 09:07:40", tz=TAIPEI)
        self.assertEqual(floor_bar(ts, 5), pd.Timestamp("2026-08-21 09:05:00", tz=TAIPEI))

    def test_aggregator_closes_previous_bucket(self) -> None:
        agg = BarAggregator(5)
        self.assertIsNone(agg.on_tick(pd.Timestamp("2026-08-21 09:01:00", tz=TAIPEI), 100.0, 1))
        self.assertIsNone(agg.on_tick(pd.Timestamp("2026-08-21 09:04:00", tz=TAIPEI), 101.0, 1))
        closed = agg.on_tick(pd.Timestamp("2026-08-21 09:05:00", tz=TAIPEI), 102.0, 1)
        self.assertIsNotNone(closed)
        self.assertEqual(closed.start, pd.Timestamp("2026-08-21 09:00:00", tz=TAIPEI))
        self.assertEqual(closed.open, 100.0)
        self.assertEqual(closed.high, 101.0)
        self.assertEqual(closed.close, 101.0)

    def test_flush_closes_stale_forming_bar(self) -> None:
        agg = BarAggregator(5)
        agg.on_tick(pd.Timestamp("2026-08-21 09:01:00", tz=TAIPEI), 100.0)
        flushed = agg.flush_if_due(pd.Timestamp("2026-08-21 09:06:00", tz=TAIPEI))
        self.assertIsNotNone(flushed)
        self.assertEqual(flushed.close, 100.0)

    def test_in_session_weekdays_only(self) -> None:
        friday = datetime(2026, 8, 21, 10, 0, tzinfo=TAIPEI)
        saturday = datetime(2026, 8, 22, 10, 0, tzinfo=TAIPEI)
        self.assertTrue(in_session(friday))
        self.assertFalse(in_session(saturday))
        self.assertFalse(in_session(datetime(2026, 8, 21, 8, 59, tzinfo=TAIPEI)))

    def test_should_run_15m_on_third_five_min_bar(self) -> None:
        self.assertTrue(
            should_run_15m(OhlcvBar(pd.Timestamp("2026-08-21 09:10:00", tz=TAIPEI), 1, 1, 1, 1))
        )
        self.assertFalse(
            should_run_15m(OhlcvBar(pd.Timestamp("2026-08-21 09:05:00", tz=TAIPEI), 1, 1, 1, 1))
        )

    def test_kbars_to_ohlcv_reads_shioaji_shape(self) -> None:
        kbars = {
            "ts": ["2026-08-21 09:00:00", "2026-08-21 09:01:00"],
            "Open": [10.0, 10.5],
            "High": [10.2, 10.6],
            "Low": [9.9, 10.4],
            "Close": [10.1, 10.5],
            "Volume": [3.0, 4.0],
        }
        df = kbars_to_ohlcv(kbars)
        self.assertEqual(len(df), 2)
        self.assertEqual(float(df.iloc[-1]["close"]), 10.5)

    def test_format_telegram_mentions_cross(self) -> None:
        df = _history_then_live([99.0, 105.0])
        hits = iter_5m_ma200_alerts(df)
        self.assertEqual(len(hits), 1)
        text = format_telegram("創見", "2451.TW", hits[0], "5m")
        self.assertIn("五分K 剛站上 MA200", text)
        self.assertIn("創見", text)
        self.assertIn("2451.TW", text)
        self.assertIn("十五分", text)

    def test_alerts_on_closed_bar_only_that_five(self) -> None:
        df = _history_then_live([99.0, 105.0])
        bar = OhlcvBar(pd.Timestamp("2026-08-21 09:05:00", tz=TAIPEI), 105, 105, 105, 105)
        hits = alerts_on_closed_bar(df, bar, tf="5m")
        self.assertEqual(len(hits), 1)
        other = OhlcvBar(pd.Timestamp("2026-08-21 09:10:00", tz=TAIPEI), 105, 105, 105, 105)
        self.assertEqual(alerts_on_closed_bar(df, other, tf="5m"), [])

    def test_upsert_bar_overwrites_same_timestamp(self) -> None:
        first = OhlcvBar(pd.Timestamp("2026-08-21 09:00:00", tz=TAIPEI), 1, 2, 1, 1.5, 10)
        frame = upsert_bar(pd.DataFrame(), first)
        second = OhlcvBar(pd.Timestamp("2026-08-21 09:00:00", tz=TAIPEI), 1, 3, 1, 2.0, 12)
        frame = upsert_bar(frame, second)
        self.assertEqual(len(frame), 1)
        self.assertEqual(float(frame.iloc[0]["close"]), 2.0)


if __name__ == "__main__":
    unittest.main()

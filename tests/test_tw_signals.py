from __future__ import annotations

import unittest

import pandas as pd

from tw.signals import (
    AlertSnapshot,
    add_moving_averages,
    close_above_ma240,
    is_intraday_entry_bar,
    is_ma240_breakout_bullish,
    latest_ma240_breakout_bullish,
    ma240_at,
    ma240_gap_pct,
)


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
        closes = [100.0] * 240 + [110.0]
        snap = is_ma240_breakout_bullish(_bars(closes))
        self.assertIsNotNone(snap)
        assert snap is not None
        self.assertGreater(snap.ma5, snap.ma10)
        self.assertGreater(snap.ma10, snap.ma20)
        self.assertGreater(snap.close, snap.ma240)
        self.assertLessEqual(snap.prev_close, snap.prev_ma240)
        self.assertEqual(snap.close, 110.0)

    def test_one_bar_above_ma240_is_enough(self) -> None:
        closes = [100.0] * 240 + [110.0]
        snap = is_ma240_breakout_bullish(_bars(closes))
        self.assertIsNotNone(snap)
        assert snap is not None
        self.assertEqual(snap.close, 110.0)
        self.assertIsNotNone(latest_ma240_breakout_bullish(_bars(closes)))

    def test_already_above_previous_bar_is_not_new_signal(self) -> None:
        closes = [100.0] * 238
        closes.extend([104.0, 105.0, 106.0, 107.0])
        snap = is_ma240_breakout_bullish(_bars(closes))
        self.assertIsNone(snap)

    def test_no_signal_without_bullish_alignment(self) -> None:
        # 最後一根剛站上 MA240，但短均不是多頭排列
        closes = [100.0] * 237 + [80.0, 80.0, 80.0, 110.0]
        snap = is_ma240_breakout_bullish(_bars(closes))
        self.assertIsNone(snap)

    def test_lookback_finds_earlier_cross_today(self) -> None:
        closes = [100.0] * 244 + [110.0] + [111.0, 112.0]
        df = _bars(closes)
        self.assertIsNone(is_ma240_breakout_bullish(df))
        snap = latest_ma240_breakout_bullish(df)
        self.assertIsNotNone(snap)
        assert snap is not None
        self.assertEqual(snap.close, 110.0)
        self.assertIsNone(is_ma240_breakout_bullish(_bars([100.0] * 240)))

    def test_until_keeps_friday_and_ignores_monday(self) -> None:
        fri = list(pd.date_range("2026-08-14 09:00", periods=246, freq="1min", tz="Asia/Taipei"))
        mon = list(pd.date_range("2026-08-17 09:00", periods=3, freq="1min", tz="Asia/Taipei"))
        closes = [100.0] * 244 + [110.0, 111.0] + [112.0, 113.0, 114.0]
        df = pd.DataFrame(
            {
                "open": closes,
                "high": [c + 0.5 for c in closes],
                "low": [c - 0.5 for c in closes],
                "close": closes,
                "volume": [1000] * len(closes),
            },
            index=fri + mon,
        )
        until = pd.Timestamp("2026-08-14 23:59:59", tz="Asia/Taipei")
        snap = latest_ma240_breakout_bullish(df, until=until)
        self.assertIsNotNone(snap)
        assert snap is not None
        self.assertEqual(snap.timestamp.date().isoformat(), "2026-08-14")
        self.assertEqual(snap.close, 110.0)

    def test_overnight_gap_is_not_intraday_cross(self) -> None:
        idx = list(pd.date_range("2026-08-14 13:00", periods=240, freq="1min", tz="Asia/Taipei"))
        idx.append(pd.Timestamp("2026-08-17 09:00", tz="Asia/Taipei"))
        closes = [100.0] * 240 + [110.0]
        df = pd.DataFrame(
            {
                "open": closes,
                "high": [c + 0.5 for c in closes],
                "low": [c - 0.5 for c in closes],
                "close": closes,
                "volume": [1000] * len(closes),
            },
            index=idx,
        )
        self.assertIsNone(is_ma240_breakout_bullish(df))
        self.assertIsNone(latest_ma240_breakout_bullish(df))

    def test_open_gap_through_ma240_is_not_entry(self) -> None:
        fri = list(pd.date_range("2026-08-14 09:46", periods=240, freq="1min", tz="Asia/Taipei"))
        mon = list(pd.date_range("2026-08-17 09:00", periods=5, freq="1min", tz="Asia/Taipei"))
        closes = [100.0] * 240 + [110.0, 111.0, 111.0, 111.0, 111.0]
        df = pd.DataFrame(
            {
                "open": closes,
                "high": [c + 0.5 for c in closes],
                "low": [c - 0.5 for c in closes],
                "close": closes,
                "volume": [1000] * len(closes),
            },
            index=fri + mon,
        )
        since = pd.Timestamp("2026-08-17", tz="Asia/Taipei")
        self.assertIsNone(latest_ma240_breakout_bullish(df, since=since))
        self.assertFalse(is_intraday_entry_bar(fri[-1], mon[0]))
        self.assertTrue(is_intraday_entry_bar(mon[0], mon[1]))

    def test_same_session_0901_cross_is_entry(self) -> None:
        fri = list(pd.date_range("2026-08-14 09:46", periods=240, freq="1min", tz="Asia/Taipei"))
        mon = list(pd.date_range("2026-08-17 09:00", periods=2, freq="1min", tz="Asia/Taipei"))
        closes = [100.0] * 240 + [100.0, 110.0]
        df = pd.DataFrame(
            {
                "open": closes,
                "high": [c + 0.5 for c in closes],
                "low": [c - 0.5 for c in closes],
                "close": closes,
                "volume": [1000] * len(closes),
            },
            index=fri + mon,
        )
        since = pd.Timestamp("2026-08-17", tz="Asia/Taipei")
        snap = latest_ma240_breakout_bullish(df, since=since)
        self.assertIsNotNone(snap)
        assert snap is not None
        self.assertEqual(snap.timestamp.strftime("%H:%M"), "09:01")
        self.assertEqual(snap.close, 110.0)

    def test_cross_after_open_is_entry(self) -> None:
        fri = list(pd.date_range("2026-08-14 09:46", periods=240, freq="1min", tz="Asia/Taipei"))
        mon = list(pd.date_range("2026-08-17 09:00", periods=11, freq="1min", tz="Asia/Taipei"))
        closes = [100.0] * 240 + [100.0] * 9 + [110.0, 111.0]
        df = pd.DataFrame(
            {
                "open": closes,
                "high": [c + 0.5 for c in closes],
                "low": [c - 0.5 for c in closes],
                "close": closes,
                "volume": [1000] * len(closes),
            },
            index=fri + mon,
        )
        since = pd.Timestamp("2026-08-17", tz="Asia/Taipei")
        snap = latest_ma240_breakout_bullish(df, since=since)
        self.assertIsNotNone(snap)
        assert snap is not None
        self.assertEqual(snap.timestamp.strftime("%H:%M"), "09:09")
        self.assertEqual(snap.close, 110.0)

    def test_first_cross_even_if_later_weave(self) -> None:
        closes = [100.0] * 238 + [110.0, 99.0, 110.0, 111.0]
        snap = latest_ma240_breakout_bullish(_bars(closes))
        self.assertIsNotNone(snap)
        assert snap is not None
        self.assertEqual(snap.close, 110.0)

    def test_first_cross_is_entry_not_later_recross(self) -> None:
        closes = [100.0] * 244 + [110.0, 111.0, 90.0, 90.0, 90.0, 120.0, 121.0]
        df = _bars(closes)
        snap = latest_ma240_breakout_bullish(df)
        self.assertIsNotNone(snap)
        assert snap is not None
        self.assertEqual(snap.close, 110.0)

    def test_moving_averages_length(self) -> None:
        df = add_moving_averages(_bars([float(i) for i in range(1, 261)]))
        self.assertTrue(pd.isna(df["ma240"].iloc[238]))
        self.assertFalse(pd.isna(df["ma240"].iloc[239]))

    def test_5m_close_above_ma240(self) -> None:
        idx = pd.date_range("2026-08-17 09:00", periods=260, freq="5min", tz="Asia/Taipei")
        closes = [100.0] * 259 + [110.0]
        df = pd.DataFrame(
            {
                "open": closes,
                "high": [c + 0.5 for c in closes],
                "low": [c - 0.5 for c in closes],
                "close": closes,
                "volume": [1000] * 260,
            },
            index=idx,
        )
        ts_1m = idx[-1] + pd.Timedelta(minutes=3)
        self.assertTrue(close_above_ma240(df, ts_1m, floor="5min"))
        self.assertTrue(close_above_ma240(df, ts_1m, floor="5min", min_gap=0.06))
        close, ma240 = ma240_at(df, ts_1m, floor="5min")
        self.assertGreater(close, ma240)
        gap = ma240_gap_pct(df, ts_1m, floor="5min")
        self.assertIsNotNone(gap)
        assert gap is not None
        self.assertGreater(gap, 0.06)

    def test_5m_hugging_ma240_fails_min_gap(self) -> None:
        idx = pd.date_range("2026-08-17 09:00", periods=260, freq="5min", tz="Asia/Taipei")
        closes = [100.0] * 259 + [101.0]
        df = pd.DataFrame(
            {
                "open": closes,
                "high": [c + 0.5 for c in closes],
                "low": [c - 0.5 for c in closes],
                "close": closes,
                "volume": [1000] * 260,
            },
            index=idx,
        )
        ts_1m = idx[-1] + pd.Timedelta(minutes=3)
        self.assertTrue(close_above_ma240(df, ts_1m, floor="5min"))
        self.assertFalse(close_above_ma240(df, ts_1m, floor="5min", min_gap=0.06))
        gap = ma240_gap_pct(df, ts_1m, floor="5min")
        self.assertIsNotNone(gap)
        assert gap is not None
        self.assertLess(gap, 0.06)

    def test_5m_close_below_ma240(self) -> None:
        idx = pd.date_range("2026-08-17 09:00", periods=260, freq="5min", tz="Asia/Taipei")
        closes = [100.0] * 259 + [90.0]
        df = pd.DataFrame(
            {
                "open": closes,
                "high": [c + 0.5 for c in closes],
                "low": [c - 0.5 for c in closes],
                "close": closes,
                "volume": [1000] * 260,
            },
            index=idx,
        )
        ts_1m = idx[-1] + pd.Timedelta(minutes=3)
        self.assertFalse(close_above_ma240(df, ts_1m, floor="5min"))

    def test_tangled_stack_still_counts_as_bullish(self) -> None:
        snap = AlertSnapshot(
            timestamp=pd.Timestamp("2026-08-14 13:20", tz="Asia/Taipei"),
            close=43.20,
            prev_close=43.10,
            ma5=43.09,
            ma10=43.06,
            ma20=43.02,
            ma240=43.15,
            prev_ma240=43.16,
        )
        self.assertTrue(snap.bullish_aligned)
        self.assertLess(snap.ma_span_pct, 0.004)


if __name__ == "__main__":
    unittest.main()

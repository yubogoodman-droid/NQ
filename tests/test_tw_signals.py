from __future__ import annotations

import unittest

import pandas as pd

from tw.signals import (
    AlertSnapshot,
    add_moving_averages,
    close_above_ma200,
    is_ma200_breakout_bullish,
    latest_ma200_breakout_bullish,
    ma200_at,
    ma200_gap_pct,
    mas_are_open,
    passes_1m_structure,
    price_popped_off_ma5,
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

    def test_lookback_finds_earlier_cross_today(self) -> None:
        closes = [100.0] * 200 + [110.0] + [111.0, 112.0]
        df = _bars(closes)
        self.assertIsNone(is_ma200_breakout_bullish(df))
        snap = latest_ma200_breakout_bullish(df)
        self.assertIsNotNone(snap)
        assert snap is not None
        self.assertEqual(snap.close, 110.0)
        self.assertIsNone(is_ma200_breakout_bullish(_bars([100.0] * 200)))

    def test_until_keeps_friday_and_ignores_monday(self) -> None:
        fri = list(pd.date_range("2026-08-14 09:00", periods=201, freq="1min", tz="Asia/Taipei"))
        mon = list(pd.date_range("2026-08-17 09:00", periods=3, freq="1min", tz="Asia/Taipei"))
        closes = [100.0] * 200 + [110.0] + [111.0, 112.0, 113.0]
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
        snap = latest_ma200_breakout_bullish(df, until=until)
        self.assertIsNotNone(snap)
        assert snap is not None
        self.assertEqual(snap.timestamp.date().isoformat(), "2026-08-14")
        self.assertEqual(snap.close, 110.0)

    def test_overnight_gap_is_not_intraday_cross(self) -> None:
        idx = list(pd.date_range("2026-08-14 13:00", periods=200, freq="1min", tz="Asia/Taipei"))
        idx.append(pd.Timestamp("2026-08-17 09:00", tz="Asia/Taipei"))
        closes = [100.0] * 200 + [110.0]
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
        self.assertIsNone(is_ma200_breakout_bullish(df))
        self.assertIsNone(latest_ma200_breakout_bullish(df))

    def test_moving_averages_length(self) -> None:
        df = add_moving_averages(_bars([float(i) for i in range(1, 221)]))
        self.assertTrue(pd.isna(df["ma200"].iloc[198]))
        self.assertFalse(pd.isna(df["ma200"].iloc[199]))

    def test_5m_close_above_ma200(self) -> None:
        idx = pd.date_range("2026-08-17 09:00", periods=220, freq="5min", tz="Asia/Taipei")
        closes = [100.0] * 219 + [110.0]
        df = pd.DataFrame(
            {
                "open": closes,
                "high": [c + 0.5 for c in closes],
                "low": [c - 0.5 for c in closes],
                "close": closes,
                "volume": [1000] * 220,
            },
            index=idx,
        )
        ts_1m = idx[-1] + pd.Timedelta(minutes=3)
        self.assertTrue(close_above_ma200(df, ts_1m, floor="5min"))
        self.assertTrue(close_above_ma200(df, ts_1m, floor="5min", min_gap=0.06))
        close, ma200 = ma200_at(df, ts_1m, floor="5min")
        self.assertGreater(close, ma200)
        gap = ma200_gap_pct(df, ts_1m, floor="5min")
        self.assertIsNotNone(gap)
        assert gap is not None
        self.assertGreater(gap, 0.06)

    def test_5m_hugging_ma200_fails_min_gap(self) -> None:
        idx = pd.date_range("2026-08-17 09:00", periods=220, freq="5min", tz="Asia/Taipei")
        closes = [100.0] * 219 + [101.0]
        df = pd.DataFrame(
            {
                "open": closes,
                "high": [c + 0.5 for c in closes],
                "low": [c - 0.5 for c in closes],
                "close": closes,
                "volume": [1000] * 220,
            },
            index=idx,
        )
        ts_1m = idx[-1] + pd.Timedelta(minutes=3)
        self.assertTrue(close_above_ma200(df, ts_1m, floor="5min"))
        self.assertFalse(close_above_ma200(df, ts_1m, floor="5min", min_gap=0.06))
        gap = ma200_gap_pct(df, ts_1m, floor="5min")
        self.assertIsNotNone(gap)
        assert gap is not None
        self.assertLess(gap, 0.06)

    def test_5m_close_below_ma200(self) -> None:
        idx = pd.date_range("2026-08-17 09:00", periods=220, freq="5min", tz="Asia/Taipei")
        closes = [100.0] * 219 + [90.0]
        df = pd.DataFrame(
            {
                "open": closes,
                "high": [c + 0.5 for c in closes],
                "low": [c - 0.5 for c in closes],
                "close": closes,
                "volume": [1000] * 220,
            },
            index=idx,
        )
        ts_1m = idx[-1] + pd.Timedelta(minutes=3)
        self.assertFalse(close_above_ma200(df, ts_1m, floor="5min"))

    def test_keeps_open_ma_stack_like_jinju(self) -> None:
        snap = AlertSnapshot(
            timestamp=pd.Timestamp("2026-08-14 09:53", tz="Asia/Taipei"),
            close=425.50,
            prev_close=424.00,
            ma5=422.90,
            ma10=421.65,
            ma20=419.65,
            ma200=424.41,
            prev_ma200=424.50,
        )
        self.assertGreater(snap.ma_span_pct, 0.005)
        self.assertTrue(mas_are_open(snap))

    def test_drops_tangled_ma_stack(self) -> None:
        snap = AlertSnapshot(
            timestamp=pd.Timestamp("2026-08-14 13:20", tz="Asia/Taipei"),
            close=43.20,
            prev_close=43.10,
            ma5=43.09,
            ma10=43.06,
            ma20=43.02,
            ma200=43.15,
            prev_ma200=43.16,
        )
        self.assertLess(snap.ma_span_pct, 0.005)
        self.assertFalse(mas_are_open(snap))
        self.assertFalse(passes_1m_structure(snap))

    def test_keeps_qunbang_style_ma5_pop(self) -> None:
        snap = AlertSnapshot(
            timestamp=pd.Timestamp("2026-08-17 09:09", tz="Asia/Taipei"),
            close=161.50,
            prev_close=159.50,
            ma5=159.70,
            ma10=159.15,
            ma20=159.03,
            ma200=160.18,
            prev_ma200=160.18,
        )
        self.assertLess(snap.ma_span_pct, 0.005)
        self.assertFalse(mas_are_open(snap))
        self.assertGreaterEqual(snap.ma5_pop_pct, 0.01)
        self.assertTrue(price_popped_off_ma5(snap))
        self.assertTrue(passes_1m_structure(snap))

    def test_keeps_xinchang_style_ma5_pop(self) -> None:
        snap = AlertSnapshot(
            timestamp=pd.Timestamp("2026-08-17 09:58", tz="Asia/Taipei"),
            close=211.00,
            prev_close=208.50,
            ma5=208.20,
            ma10=207.95,
            ma20=207.65,
            ma200=209.68,
            prev_ma200=209.69,
        )
        self.assertLess(snap.ma_span_pct, 0.005)
        self.assertFalse(mas_are_open(snap))
        self.assertGreaterEqual(snap.ma5_pop_pct, 0.01)
        self.assertTrue(passes_1m_structure(snap))


if __name__ == "__main__":
    unittest.main()

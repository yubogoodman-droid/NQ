from __future__ import annotations

import tempfile
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd

from tw.backtest_5m import BacktestConfig, BacktestHit, BacktestResult, DayUniverse, run_5m_backtest
from tw.kline import resample_ohlcv
from tw.ranking import RankedStock
from tw.forward import hour_later, summarize_hour_later
from tw.report import _session_tick_labels, save_backtest_html, weekday_zh
from tw.signals import iter_15m_ma240_alerts, iter_5m_ma240_alerts

TAIPEI = ZoneInfo("Asia/Taipei")


def _session_index(days: list[date], bars_per_day: int = 54) -> pd.DatetimeIndex:
    stamps: list[datetime] = []
    for day in days:
        start = datetime(day.year, day.month, day.day, 9, 0, tzinfo=TAIPEI)
        stamps.extend(start + timedelta(minutes=5 * i) for i in range(bars_per_day))
    return pd.DatetimeIndex(stamps)


def _ohlcv(closes: list[float], index: pd.DatetimeIndex) -> pd.DataFrame:
    close = pd.Series(closes, index=index, dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "volume": 1000.0,
        }
    )


def _daily_ohlcv(end: date, days: int = 220, last_close: float = 105.0) -> pd.DataFrame:
    idx = pd.bdate_range(end=end, periods=days, tz=TAIPEI)
    close = pd.Series([100.0 + 0.02 * i for i in range(days)], index=idx, dtype=float)
    close.iloc[-1] = last_close
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 1000.0,
        }
    )


def _weekdays_before(end: date, n: int) -> list[date]:
    days: list[date] = []
    current = end
    while len(days) < n:
        if current.weekday() < 5:
            days.append(current)
        current -= timedelta(days=1)
    days.reverse()
    return days


def _history_then_live(live_closes: list[float], live_day: date = date(2026, 8, 21)) -> pd.DataFrame:
    hist_days = _weekdays_before(live_day - timedelta(days=1), 14)
    hist = _ohlcv([100.0] * (54 * len(hist_days)), _session_index(hist_days))
    live_index = _session_index([live_day])[: len(live_closes)]
    live = _ohlcv(live_closes, live_index)
    return pd.concat([hist, live])


class FiveMinSignalTests(unittest.TestCase):
    def test_alerts_on_cross_with_bullish_mas(self) -> None:
        df = _history_then_live([99.0, 105.0])
        hits = iter_5m_ma240_alerts(df)
        self.assertEqual(len(hits), 1)
        hit = hits[0]
        self.assertTrue(hit.bullish_aligned)
        self.assertTrue(hit.mas_rising)
        self.assertTrue(hit.crossed_above_ma240)
        self.assertTrue(hit.close_above_all_mas)
        self.assertGreater(hit.close, hit.ma5)
        self.assertGreater(hit.close, hit.ma240)
        self.assertLessEqual(hit.prev_close, hit.prev_ma240)
        self.assertEqual(hit.timestamp, pd.Timestamp(datetime(2026, 8, 21, 9, 5, tzinfo=TAIPEI)))

    def test_does_not_repeat_while_staying_above(self) -> None:
        df = _history_then_live([99.0] + [105.0] * 8)
        hits = iter_5m_ma240_alerts(df)
        self.assertEqual(len(hits), 1)

    def test_counts_gap_up_first_bar_of_the_day(self) -> None:
        df = _history_then_live([105.0, 105.0])
        hits = iter_5m_ma240_alerts(df)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].timestamp, pd.Timestamp(datetime(2026, 8, 21, 9, 0, tzinfo=TAIPEI)))

    def test_since_until_keeps_only_that_session(self) -> None:
        df = _history_then_live([99.0, 105.0])
        since = pd.Timestamp(date(2026, 8, 20), tz=TAIPEI)
        until = since + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        self.assertEqual(iter_5m_ma240_alerts(df, since=since, until=until), [])
        since2 = pd.Timestamp(date(2026, 8, 21), tz=TAIPEI)
        until2 = since2 + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        hits = iter_5m_ma240_alerts(df, since=since2, until=until2)
        self.assertEqual(len(hits), 1)

    def test_rejects_close_below_ma5(self) -> None:
        df = _history_then_live([110.0] * 5 + [99.0, 100.5])
        hits = iter_5m_ma240_alerts(df)
        self.assertFalse(any(abs(h.close - 100.5) < 1e-9 for h in hits))

    def test_requires_bullish_ribbon(self) -> None:
        hist_days = [date(2026, 8, 17), date(2026, 8, 18), date(2026, 8, 19), date(2026, 8, 20)]
        hist = _ohlcv([110.0] * (54 * 4), _session_index(hist_days))
        live = _ohlcv(
            [90.0] * 10 + [111.0],
            _session_index([date(2026, 8, 21)])[:11],
        )
        hits = iter_5m_ma240_alerts(pd.concat([hist, live]))
        self.assertEqual(hits, [])


class ShortFiveMinSignalTests(unittest.TestCase):
    def test_alerts_on_cross_with_bearish_mas(self) -> None:
        df = _history_then_live([101.0, 95.0])
        hits = iter_5m_ma240_alerts(df, side="short")
        self.assertEqual(len(hits), 1)
        hit = hits[0]
        self.assertEqual(hit.side, "short")
        self.assertTrue(hit.bearish_aligned)
        self.assertTrue(hit.mas_falling)
        self.assertTrue(hit.ribbon_down)
        self.assertTrue(hit.crossed_below_ma240)
        self.assertTrue(hit.close_below_all_mas)
        self.assertLess(hit.close, hit.ma5)
        self.assertLess(hit.close, hit.ma240)
        self.assertGreaterEqual(hit.prev_close, hit.prev_ma240)
        self.assertEqual(hit.timestamp, pd.Timestamp(datetime(2026, 8, 21, 9, 5, tzinfo=TAIPEI)))

    def test_does_not_repeat_while_staying_below(self) -> None:
        df = _history_then_live([101.0] + [95.0] * 8)
        hits = iter_5m_ma240_alerts(df, side="short")
        self.assertEqual(len(hits), 1)

    def test_counts_gap_down_first_bar_of_the_day(self) -> None:
        df = _history_then_live([95.0, 95.0])
        hits = iter_5m_ma240_alerts(df, side="short")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].timestamp, pd.Timestamp(datetime(2026, 8, 21, 9, 0, tzinfo=TAIPEI)))

    def test_long_cross_is_not_a_short_alert(self) -> None:
        df = _history_then_live([100.0, 105.0])
        self.assertEqual(iter_5m_ma240_alerts(df, side="short"), [])
        both = iter_5m_ma240_alerts(df, side="both")
        self.assertEqual(len(both), 1)
        self.assertEqual(both[0].side, "long")

    def test_requires_bearish_ribbon(self) -> None:
        hist_days = [date(2026, 8, 17), date(2026, 8, 18), date(2026, 8, 19), date(2026, 8, 20)]
        hist = _ohlcv([90.0] * (54 * 4), _session_index(hist_days))
        live = _ohlcv(
            [110.0] * 10 + [89.0],
            _session_index([date(2026, 8, 21)])[:11],
        )
        hits = iter_5m_ma240_alerts(pd.concat([hist, live]), side="short")
        self.assertEqual(hits, [])


class FiveMinBacktestTests(unittest.TestCase):
    def test_run_scan_uses_daily_universe_and_5m_alerts(self) -> None:
        df = _history_then_live([99.0, 105.0])
        stock = RankedStock(1, "2408.TW", "南亞科", 105.0, 1.0, 1.0, 100, 1e9, "TAI")
        universe = DayUniverse(
            day=date(2026, 8, 21),
            rank_time="test",
            universe=[stock],
            candidates=[stock],
            price_dropped=0,
            etf_dropped=0,
            financial_dropped=0,
            telecom_dropped=0,
        )

        def fake_universes(cfg, as_of, sess):
            return [universe]

        with (
            patch("tw.backtest_5m._load_session_universes", side_effect=fake_universes),
            patch("tw.backtest_5m.fetch_bars_many", return_value={"2408.TW": df}),
        ):
            result = run_5m_backtest(BacktestConfig(days=1, today=date(2026, 8, 21)))
        self.assertEqual(len(result.hits), 1)
        self.assertEqual(result.hits[0].stock.symbol, "2408.TW")
        self.assertEqual(result.hits[0].day, date(2026, 8, 21))

    def test_run_scan_short_side_uses_down_cross(self) -> None:
        df = _history_then_live([101.0, 95.0])
        stock = RankedStock(1, "2408.TW", "南亞科", 95.0, -1.0, -1.0, 100, 1e9, "TAI")
        universe = DayUniverse(
            day=date(2026, 8, 21),
            rank_time="test",
            universe=[stock],
            candidates=[stock],
            price_dropped=0,
            etf_dropped=0,
            financial_dropped=0,
            telecom_dropped=0,
        )

        def fake_universes(cfg, as_of, sess):
            return [universe]

        with (
            patch("tw.backtest_5m._load_session_universes", side_effect=fake_universes),
            patch("tw.backtest_5m.fetch_bars_many", return_value={"2408.TW": df}),
        ):
            result = run_5m_backtest(
                BacktestConfig(days=1, today=date(2026, 8, 21), side="short")
            )
        self.assertEqual(len(result.hits), 1)
        self.assertEqual(result.side, "short")
        self.assertEqual(result.hits[0].snapshot.side, "short")


class ReportTests(unittest.TestCase):
    def test_html_contains_hit(self) -> None:
        df = _history_then_live([99.0, 105.0])
        hits = iter_5m_ma240_alerts(df)
        stock = RankedStock(1, "2408.TW", "南亞科", 105.0, 1.0, 1.0, 100, 1e9, "TAI")
        result = BacktestResult(
            scanned_at=datetime(2026, 8, 21, 17, 0, tzinfo=TAIPEI),
            days=[date(2026, 8, 21)],
            universes={
                date(2026, 8, 21): DayUniverse(
                    day=date(2026, 8, 21),
                    rank_time="t",
                    universe=[stock],
                    candidates=[stock],
                    price_dropped=2,
                    etf_dropped=1,
                    financial_dropped=0,
                    telecom_dropped=0,
                )
            },
            hits=[
                BacktestHit(
                    day=date(2026, 8, 21),
                    stock=stock,
                    snapshot=hits[0],
                    frame=df,
                    daily=_daily_ohlcv(date(2026, 8, 21)),
                )
            ],
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = save_backtest_html(result, Path(tmp) / "out.html")
            text = path.read_text(encoding="utf-8")
            pngs = list(Path(tmp).joinpath("charts/out").glob("*.png"))
            self.assertEqual(len(pngs), 1)
            self.assertTrue(pngs[0].name.endswith("-stack.png"))
            from PIL import Image

            with Image.open(pngs[0]) as im:
                self.assertGreater(im.height, 1400)
        self.assertIn("南亞科", text)
        self.assertIn("十五分K", text)
        self.assertIn("日K", text)
        self.assertIn("日K / MA5 10 20 60 240", text)
        self.assertIn("最下＝日K", text)
        self.assertIn("上＝五分K", text)
        self.assertIn("成交額前 200", text)
        self.assertIn("股價 &lt; 500", text)
        self.assertIn("一小時後", text)
        self.assertIn("進場後一小時勝率", text)
        self.assertIn("個交易日共通知", text)
        self.assertNotIn("小時 MA20 不下彎", text)
        self.assertNotIn("十五分K已在 MA240 上至少半小時", text)
        self.assertEqual(text.count("<img "), 1)
        self.assertEqual(weekday_zh(date(2026, 8, 21)), "週五")

    def test_axis_labels_mark_the_next_session_after_friday(self) -> None:
        friday = pd.date_range("2026-08-14 12:00", periods=3, freq="h", tz=TAIPEI)
        monday = pd.date_range("2026-08-17 09:00", periods=3, freq="h", tz=TAIPEI)
        idx = friday.append(monday)
        ticks, labels = _session_tick_labels(idx)
        self.assertIn(3, ticks)
        self.assertTrue(any("08/14" in lab for lab in labels))
        self.assertTrue(any("08/17" in lab for lab in labels))
        self.assertFalse(any("08/15" in lab or "08/16" in lab for lab in labels))


class HourLaterTests(unittest.TestCase):
    def test_reads_the_same_session_bar_one_hour_later(self) -> None:
        idx = _session_index([date(2026, 8, 21)])
        closes = [100.0] * len(idx)
        closes[12] = 103.0
        df = _ohlcv(closes, idx)
        move = hour_later(df, idx[0], 100.0)
        self.assertIsNotNone(move)
        self.assertEqual(move.later, 103.0)
        self.assertTrue(move.win)
        self.assertAlmostEqual(move.ret_pct, 3.0)

    def test_short_win_when_price_drops(self) -> None:
        idx = _session_index([date(2026, 8, 21)])
        closes = [100.0] * len(idx)
        closes[12] = 97.0
        df = _ohlcv(closes, idx)
        move = hour_later(df, idx[0], 100.0, side="short")
        self.assertIsNotNone(move)
        self.assertEqual(move.later, 97.0)
        self.assertTrue(move.win)
        self.assertAlmostEqual(move.ret_pct, 3.0)

    def test_skips_when_the_session_has_no_full_hour(self) -> None:
        idx = _session_index([date(2026, 8, 21)])
        df = _ohlcv([100.0] * len(idx), idx)
        move = hour_later(df, idx[-5], 100.0)
        self.assertIsNone(move)

    def test_summary_counts_wins_and_short_sessions(self) -> None:
        df = _history_then_live([99.0, 105.0] + [106.0] * 20)
        hits = iter_5m_ma240_alerts(df)
        self.assertEqual(len(hits), 1)
        stock = RankedStock(1, "2408.TW", "南亞科", 105.0, 1.0, 1.0, 100, 1e9, "TAI")
        scored = BacktestHit(day=date(2026, 8, 21), stock=stock, snapshot=hits[0], frame=df)
        short = BacktestHit(
            day=date(2026, 8, 21),
            stock=stock,
            snapshot=hits[0],
            frame=_history_then_live([99.0, 105.0]),
        )
        stats = summarize_hour_later([scored, short])
        self.assertEqual(stats.n_scored, 1)
        self.assertEqual(stats.n_short, 1)
        self.assertEqual(stats.wins, 1)
        self.assertAlmostEqual(stats.win_rate, 100.0)


class FifteenMinSignalTests(unittest.TestCase):
    def test_alerts_on_15m_cross_while_ribbon_is_fanned(self) -> None:
        df = _history_then_live([99.0, 99.0, 99.0, 105.0, 105.0, 105.0])
        hits = iter_15m_ma240_alerts(df)
        self.assertEqual(len(hits), 1)
        hit = hits[0]
        self.assertTrue(hit.ribbon_fanned)
        self.assertTrue(hit.crossed_above_ma240)
        self.assertTrue(hit.close_above_all_mas)
        self.assertGreater(hit.close, hit.ma240)
        self.assertLessEqual(hit.prev_close, hit.prev_ma240)
        self.assertEqual(hit.timestamp, pd.Timestamp(datetime(2026, 8, 21, 9, 25, tzinfo=TAIPEI)))

    def test_does_not_repeat_while_staying_above(self) -> None:
        df = _history_then_live([99.0, 99.0, 99.0] + [105.0] * 9)
        hits = iter_15m_ma240_alerts(df)
        self.assertEqual(len(hits), 1)

    def test_counts_first_15m_of_the_day(self) -> None:
        df = _history_then_live([99.0, 110.0, 110.0])
        hits = iter_15m_ma240_alerts(df)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].timestamp, pd.Timestamp(datetime(2026, 8, 21, 9, 10, tzinfo=TAIPEI)))

    def test_alerts_on_15m_down_cross(self) -> None:
        df = _history_then_live([101.0, 101.0, 101.0, 95.0, 95.0, 95.0])
        hits = iter_15m_ma240_alerts(df, side="short")
        self.assertEqual(len(hits), 1)
        hit = hits[0]
        self.assertEqual(hit.side, "short")
        self.assertTrue(hit.ribbon_down)
        self.assertTrue(hit.crossed_below_ma240)
        self.assertTrue(hit.close_below_all_mas)
        self.assertEqual(hit.timestamp, pd.Timestamp(datetime(2026, 8, 21, 9, 25, tzinfo=TAIPEI)))

    def test_counts_first_15m_gap_down(self) -> None:
        df = _history_then_live([101.0, 90.0, 90.0])
        hits = iter_15m_ma240_alerts(df, side="short")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].timestamp, pd.Timestamp(datetime(2026, 8, 21, 9, 10, tzinfo=TAIPEI)))


class FifteenMinReportTests(unittest.TestCase):
    def test_html_uses_15m_copy(self) -> None:
        df = _history_then_live([99.0, 99.0, 99.0, 105.0, 105.0, 105.0])
        hits = iter_15m_ma240_alerts(df)
        stock = RankedStock(1, "2408.TW", "南亞科", 105.0, 1.0, 1.0, 100, 1e9, "TAI")
        result = BacktestResult(
            scanned_at=datetime(2026, 8, 21, 17, 0, tzinfo=TAIPEI),
            days=[date(2026, 8, 21)],
            universes={
                date(2026, 8, 21): DayUniverse(
                    day=date(2026, 8, 21),
                    rank_time="t",
                    universe=[stock],
                    candidates=[stock],
                    price_dropped=0,
                    etf_dropped=0,
                    financial_dropped=0,
                    telecom_dropped=0,
                )
            },
            hits=[
                BacktestHit(
                    day=date(2026, 8, 21),
                    stock=stock,
                    snapshot=hits[0],
                    frame=df,
                    daily=_daily_ohlcv(date(2026, 8, 21)),
                    signal_tf="15m",
                )
            ],
            signal_tf="15m",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = save_backtest_html(result, Path(tmp) / "out.html")
            text = path.read_text(encoding="utf-8")
        self.assertIn("台股十五分K回測", text)
        self.assertIn("十五分站上時間", text)
        self.assertIn("當根收盤剛站上十五分 MA240", text)
        self.assertNotIn("十五分K已在 MA240 上至少半小時", text)
        self.assertNotIn("小時K也要在 MA5／10／20 之上", text)


class ShortReportTests(unittest.TestCase):
    def test_html_uses_short_copy(self) -> None:
        df = _history_then_live([101.0, 95.0])
        hits = iter_5m_ma240_alerts(df, side="short")
        stock = RankedStock(1, "2408.TW", "南亞科", 95.0, -1.0, -1.0, 100, 1e9, "TAI")
        result = BacktestResult(
            scanned_at=datetime(2026, 8, 21, 17, 0, tzinfo=TAIPEI),
            days=[date(2026, 8, 21)],
            universes={
                date(2026, 8, 21): DayUniverse(
                    day=date(2026, 8, 21),
                    rank_time="t",
                    universe=[stock],
                    candidates=[stock],
                    price_dropped=2,
                    etf_dropped=1,
                    financial_dropped=0,
                    telecom_dropped=0,
                )
            },
            hits=[
                BacktestHit(
                    day=date(2026, 8, 21),
                    stock=stock,
                    snapshot=hits[0],
                    frame=df,
                    daily=_daily_ohlcv(date(2026, 8, 21), last_close=95.0),
                )
            ],
            side="short",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = save_backtest_html(result, Path(tmp) / "out.html")
            text = path.read_text(encoding="utf-8")
        self.assertIn("台股空方五分K回測", text)
        self.assertIn("五分跌破時間", text)
        self.assertIn("當根收盤剛跌破五分 MA240", text)
        self.assertIn("MA5 &lt; MA10 &lt; MA20 且往下", text)
        self.assertIn("空方以一小時後價格下跌為贏", text)


class ResampleTests(unittest.TestCase):
    def test_three_5m_bars_make_one_15m(self) -> None:
        idx = pd.date_range("2026-08-21 09:00", periods=3, freq="5min", tz=TAIPEI)
        df = pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0],
                "high": [101.0, 103.0, 104.0],
                "low": [99.0, 100.0, 101.0],
                "close": [101.0, 102.0, 103.0],
                "volume": [1.0, 2.0, 3.0],
            },
            index=idx,
        )
        out = resample_ohlcv(df)
        self.assertEqual(len(out), 1)
        row = out.iloc[0]
        self.assertEqual(float(row["open"]), 100.0)
        self.assertEqual(float(row["high"]), 104.0)
        self.assertEqual(float(row["low"]), 99.0)
        self.assertEqual(float(row["close"]), 103.0)
        self.assertEqual(float(row["volume"]), 6.0)

    def test_twelve_5m_bars_make_one_hour(self) -> None:
        idx = pd.date_range("2026-08-21 09:00", periods=12, freq="5min", tz=TAIPEI)
        df = pd.DataFrame(
            {
                "open": [100.0] * 12,
                "high": [101.0] * 12,
                "low": [99.0] * 12,
                "close": list(range(100, 112)),
                "volume": [1.0] * 12,
            },
            index=idx,
        )
        out = resample_ohlcv(df, "1h")
        self.assertEqual(len(out), 1)
        self.assertEqual(float(out.iloc[0]["open"]), 100.0)
        self.assertEqual(float(out.iloc[0]["close"]), 111.0)
        self.assertEqual(float(out.iloc[0]["volume"]), 12.0)


if __name__ == "__main__":
    unittest.main()

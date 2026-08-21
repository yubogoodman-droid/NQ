from __future__ import annotations

import tempfile
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd

from tw.backtest_5m import BacktestConfig, BacktestHit, BacktestResult, DayUniverse, run_5m_backtest
from tw.ranking import RankedStock
from tw.report import save_backtest_html, weekday_zh
from tw.signals import iter_5m_ma200_alerts

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


def _history_then_live(live_closes: list[float], live_day: date = date(2026, 8, 21)) -> pd.DataFrame:
    hist_days = [date(2026, 8, 17), date(2026, 8, 18), date(2026, 8, 19), date(2026, 8, 20)]
    hist = _ohlcv([100.0] * (54 * 4), _session_index(hist_days))
    live_index = _session_index([live_day])[: len(live_closes)]
    live = _ohlcv(live_closes, live_index)
    return pd.concat([hist, live])


class FiveMinSignalTests(unittest.TestCase):
    def test_alerts_on_cross_while_ribbon_is_bullish(self) -> None:
        df = _history_then_live([100.0, 100.4, 100.8, 101.2])
        hits = iter_5m_ma200_alerts(df)
        self.assertEqual(len(hits), 1)
        hit = hits[0]
        self.assertTrue(hit.bullish_aligned)
        self.assertTrue(hit.crossed_above_ma200)
        self.assertTrue(hit.close_above_all_mas)
        self.assertGreater(hit.close, hit.ma5)
        self.assertGreater(hit.close, hit.ma200)
        self.assertLessEqual(hit.prev_close, hit.prev_ma200)
        self.assertEqual(hit.timestamp, pd.Timestamp(datetime(2026, 8, 21, 9, 5, tzinfo=TAIPEI)))

    def test_does_not_repeat_while_staying_above(self) -> None:
        df = _history_then_live([100.0] + [101.0] * 8)
        hits = iter_5m_ma200_alerts(df)
        self.assertEqual(len(hits), 1)

    def test_skips_first_bar_of_the_day(self) -> None:
        df = _history_then_live([102.0, 102.0])
        hits = iter_5m_ma200_alerts(df)
        self.assertEqual(hits, [])

    def test_since_until_keeps_only_that_session(self) -> None:
        df = _history_then_live([100.0, 101.5])
        since = pd.Timestamp(date(2026, 8, 20), tz=TAIPEI)
        until = since + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        self.assertEqual(iter_5m_ma200_alerts(df, since=since, until=until), [])
        since2 = pd.Timestamp(date(2026, 8, 21), tz=TAIPEI)
        until2 = since2 + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        hits = iter_5m_ma200_alerts(df, since=since2, until=until2)
        self.assertEqual(len(hits), 1)

    def test_rejects_close_below_ma5(self) -> None:
        df = _history_then_live([110.0] * 5 + [99.0, 100.5])
        hits = iter_5m_ma200_alerts(df)
        self.assertEqual(hits, [])

    def test_requires_bullish_ribbon(self) -> None:
        hist_days = [date(2026, 8, 17), date(2026, 8, 18), date(2026, 8, 19), date(2026, 8, 20)]
        hist = _ohlcv([110.0] * (54 * 4), _session_index(hist_days))
        live = _ohlcv(
            [90.0] * 10 + [111.0],
            _session_index([date(2026, 8, 21)])[:11],
        )
        hits = iter_5m_ma200_alerts(pd.concat([hist, live]))
        self.assertEqual(hits, [])


class FiveMinBacktestTests(unittest.TestCase):
    def test_run_scan_uses_daily_universe_and_5m_alerts(self) -> None:
        df = _history_then_live([100.0, 101.0])
        stock = RankedStock(1, "2408.TW", "南亞科", 101.0, 1.0, 1.0, 100, 1e9, "TAI")
        universe = DayUniverse(
            day=date(2026, 8, 21),
            rank_time="test",
            universe=[stock],
            candidates=[stock],
            price_dropped=0,
            etf_dropped=0,
            financial_dropped=0,
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


class ReportTests(unittest.TestCase):
    def test_html_contains_hit(self) -> None:
        df = _history_then_live([100.0, 101.0])
        hits = iter_5m_ma200_alerts(df)
        stock = RankedStock(1, "2408.TW", "南亞科", 101.0, 1.0, 1.0, 100, 1e9, "TAI")
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
                )
            },
            hits=[
                BacktestHit(
                    day=date(2026, 8, 21),
                    stock=stock,
                    snapshot=hits[0],
                    frame=df,
                )
            ],
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = save_backtest_html(result, Path(tmp) / "out.html")
            text = path.read_text(encoding="utf-8")
        self.assertIn("南亞科", text)
        self.assertIn("五分K", text)
        self.assertEqual(weekday_zh(date(2026, 8, 21)), "週五")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import tempfile
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd

from tw.backtest_5m import (
    BacktestConfig,
    BacktestHit,
    BacktestResult,
    DayUniverse,
    run_5m_short_backtest,
    summarize_forwards,
)
from tw.kline import resample_ohlcv
from tw.notify import format_hit_message
from tw.ranking import RankedStock
from tw.report import save_backtest_html, weekday_zh
from tw.signals import iter_5m_ma200_short_alerts
from tw.watch import market_open

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


class FiveMinShortSignalTests(unittest.TestCase):
    def test_alerts_on_bearish_breakdown(self) -> None:
        df = _history_then_live([100.0, 99.2])
        hits = iter_5m_ma200_short_alerts(df)
        self.assertEqual(len(hits), 1)
        hit = hits[0]
        self.assertTrue(hit.bearish_aligned)
        self.assertTrue(hit.crossed_below_ma200)
        self.assertTrue(hit.hourly_close_below_ma20)
        self.assertIsNotNone(hit.h1_close)
        self.assertLess(hit.h1_close, hit.h1_ma20)
        self.assertLess(hit.close, hit.ma200)
        self.assertGreaterEqual(hit.prev_close, hit.prev_ma200)
        self.assertLess(hit.ma5, hit.ma10)
        self.assertLess(hit.ma10, hit.ma20)
        self.assertEqual(hit.timestamp, pd.Timestamp(datetime(2026, 8, 21, 9, 5, tzinfo=TAIPEI)))

    def test_does_not_repeat_while_staying_below(self) -> None:
        df = _history_then_live([100.0, 99.2, 98.5, 98.0])
        hits = iter_5m_ma200_short_alerts(df)
        self.assertEqual(len(hits), 1)

    def test_skips_first_bar_of_the_day(self) -> None:
        df = _history_then_live([99.2, 98.8])
        hits = iter_5m_ma200_short_alerts(df)
        self.assertEqual(hits, [])

    def test_since_until_keeps_only_that_session(self) -> None:
        df = _history_then_live([100.0, 99.2])
        since = pd.Timestamp(date(2026, 8, 20), tz=TAIPEI)
        until = since + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        self.assertEqual(iter_5m_ma200_short_alerts(df, since=since, until=until), [])
        since2 = pd.Timestamp(date(2026, 8, 21), tz=TAIPEI)
        until2 = since2 + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        hits = iter_5m_ma200_short_alerts(df, since=since2, until=until2)
        self.assertEqual(len(hits), 1)

    def test_rejects_when_hourly_close_above_ma20(self) -> None:
        hist_days = [date(2026, 8, 17), date(2026, 8, 18), date(2026, 8, 19), date(2026, 8, 20)]
        idx = _session_index(hist_days)
        closes = [80.0] * 24 + [100.0] * (len(idx) - 24)
        hist = _ohlcv(closes, idx)
        live = _ohlcv([100.0, 99.2], _session_index([date(2026, 8, 21)])[:2])
        hits = iter_5m_ma200_short_alerts(pd.concat([hist, live]))
        self.assertEqual(hits, [])

    def test_requires_bearish_ribbon(self) -> None:
        hist_days = [date(2026, 8, 17), date(2026, 8, 18), date(2026, 8, 19), date(2026, 8, 20)]
        hist = _ohlcv([90.0] * (54 * 4), _session_index(hist_days))
        live = _ohlcv(
            [90.0] * 10 + [120.0, 89.0],
            _session_index([date(2026, 8, 21)])[:12],
        )
        hits = iter_5m_ma200_short_alerts(pd.concat([hist, live]))
        self.assertEqual(hits, [])

    def test_latest_only_keeps_last_hit(self) -> None:
        df = _history_then_live([100.0, 99.2] + [100.5, 99.1])
        hits = iter_5m_ma200_short_alerts(df, latest_only=True)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].timestamp, pd.Timestamp(datetime(2026, 8, 21, 9, 15, tzinfo=TAIPEI)))


class FiveMinShortBacktestTests(unittest.TestCase):
    def test_run_scan_uses_daily_universe_and_5m_alerts(self) -> None:
        df = _history_then_live([100.0, 99.2, 98.6, 98.0, 97.4])
        stock = RankedStock(1, "2408.TW", "南亞科", 97.4, -2.6, -2.6, 100, 1e9, "TAI")
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
            result = run_5m_short_backtest(BacktestConfig(days=1, today=date(2026, 8, 21)))
        self.assertEqual(len(result.hits), 1)
        self.assertEqual(result.hits[0].stock.symbol, "2408.TW")
        self.assertEqual(result.hits[0].day, date(2026, 8, 21))
        self.assertIn(3, result.hits[0].forwards)
        self.assertGreater(result.hits[0].forwards[3].pnl_pct, 0)
        stats = summarize_forwards(result.hits)
        self.assertGreater(stats["h3"]["n"], 0)


class ReportTests(unittest.TestCase):
    def test_html_contains_hit(self) -> None:
        df = _history_then_live([100.0, 99.2, 98.6, 98.0])
        hits = iter_5m_ma200_short_alerts(df)
        stock = RankedStock(1, "2408.TW", "南亞科", 99.2, -0.8, -0.8, 100, 1e9, "TAI")
        loc = df.index.get_indexer([hits[0].timestamp], method="nearest")[0]
        from tw.backtest_5m import _eod_move, _forward_moves

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
                    forwards=_forward_moves(df, loc, hits[0].close),
                    eod=_eod_move(df, loc, hits[0].close),
                )
            ],
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = save_backtest_html(result, Path(tmp) / "out.html")
            text = path.read_text(encoding="utf-8")
            pngs = list(Path(tmp).joinpath("charts/out").glob("*.png"))
            self.assertEqual(len(pngs), 1)
            self.assertTrue(pngs[0].name.endswith("-5m15m.png"))
            from PIL import Image

            with Image.open(pngs[0]) as im:
                self.assertGreater(im.height, 700)
        self.assertIn("南亞科", text)
        self.assertIn("空頭排列", text)
        self.assertIn("跌破", text)
        self.assertIn("小時K", text)
        self.assertIn("十五分K", text)
        self.assertEqual(text.count("<img "), 1)
        self.assertEqual(weekday_zh(date(2026, 8, 21)), "週五")


class NotifyTests(unittest.TestCase):
    def test_format_mentions_breakdown(self) -> None:
        df = _history_then_live([100.0, 99.2])
        snap = iter_5m_ma200_short_alerts(df)[0]
        stock = RankedStock(1, "2408.TW", "南亞科", 99.2, -0.8, -0.8, 100, 1e9, "TAI")
        title, body = format_hit_message([(stock, snap)])
        self.assertIn("跌破MA200", title)
        self.assertIn("南亞科", body)
        self.assertIn("< MA200", body)
        self.assertIn("MA5", body)
        self.assertIn("小時K", body)


class MarketHoursTests(unittest.TestCase):
    def test_weekday_session(self) -> None:
        open_bar = datetime(2026, 8, 21, 9, 5, tzinfo=TAIPEI)
        closed = datetime(2026, 8, 21, 14, 0, tzinfo=TAIPEI)
        weekend = datetime(2026, 8, 22, 10, 0, tzinfo=TAIPEI)
        self.assertTrue(market_open(open_bar))
        self.assertFalse(market_open(closed))
        self.assertFalse(market_open(weekend))


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


if __name__ == "__main__":
    unittest.main()

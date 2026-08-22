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
    def test_alerts_on_cross_while_ribbon_is_fanned(self) -> None:
        df = _history_then_live([99.0, 105.0])
        hits = iter_5m_ma200_alerts(df)
        self.assertEqual(len(hits), 1)
        hit = hits[0]
        self.assertTrue(hit.bullish_aligned)
        self.assertTrue(hit.ribbon_fanned)
        self.assertTrue(hit.crossed_above_ma200)
        self.assertTrue(hit.close_above_all_mas)
        self.assertTrue(hit.hourly_close_above_ma20)
        self.assertTrue(hit.close_above_15m_mas)
        self.assertIsNotNone(hit.h1_close)
        self.assertGreater(hit.h1_close, hit.h1_ma20)
        self.assertGreater(hit.m15_close, hit.m15_ma5)
        self.assertGreater(hit.m15_close, hit.m15_ma10)
        self.assertGreater(hit.m15_close, hit.m15_ma20)
        self.assertGreaterEqual(hit.ribbon_fan_pct, 0.50)
        self.assertGreater(hit.close, hit.ma5)
        self.assertGreater(hit.close, hit.ma200)
        self.assertLessEqual(hit.prev_close, hit.prev_ma200)
        self.assertEqual(hit.timestamp, pd.Timestamp(datetime(2026, 8, 21, 9, 5, tzinfo=TAIPEI)))

    def test_does_not_repeat_while_staying_above(self) -> None:
        df = _history_then_live([99.0] + [105.0] * 8)
        hits = iter_5m_ma200_alerts(df)
        self.assertEqual(len(hits), 1)

    def test_skips_first_bar_of_the_day(self) -> None:
        df = _history_then_live([102.0, 102.0])
        hits = iter_5m_ma200_alerts(df)
        self.assertEqual(hits, [])

    def test_since_until_keeps_only_that_session(self) -> None:
        df = _history_then_live([99.0, 105.0])
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

    def test_rejects_tangled_ribbon(self) -> None:
        df = _history_then_live([100.0, 100.4, 100.8, 101.2])
        hits = iter_5m_ma200_alerts(df)
        self.assertEqual(hits, [])

    def test_rejects_when_hourly_close_below_ma20(self) -> None:
        hist_days = [date(2026, 8, 17), date(2026, 8, 18), date(2026, 8, 19), date(2026, 8, 20)]
        idx = _session_index(hist_days)
        closes = [200.0] * 24 + [100.0] * (len(idx) - 24)
        hist = _ohlcv(closes, idx)
        live = _ohlcv([99.0, 105.0], _session_index([date(2026, 8, 21)])[:2])
        hits = iter_5m_ma200_alerts(pd.concat([hist, live]))
        self.assertEqual(hits, [])

    def test_rejects_when_close_below_15m_mas(self) -> None:
        hist_days = [date(2026, 8, 17), date(2026, 8, 18), date(2026, 8, 19), date(2026, 8, 20)]
        idx = _session_index(hist_days)
        closes = [100.0] * len(idx)
        # 一根還在最近 20 根十五分K裡的高K，把十五分 MA20 抬高，但不破壞五分剛站上。
        for i in range(159, 162):
            closes[i] = 400.0
        hist = _ohlcv(closes, idx)
        live = _ohlcv([99.0, 105.0], _session_index([date(2026, 8, 21)])[:2])
        hits = iter_5m_ma200_alerts(pd.concat([hist, live]))
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


class ReportTests(unittest.TestCase):
    def test_html_contains_hit(self) -> None:
        df = _history_then_live([99.0, 105.0])
        hits = iter_5m_ma200_alerts(df)
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
                )
            ],
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = save_backtest_html(result, Path(tmp) / "out.html")
            text = path.read_text(encoding="utf-8")
            pngs = list(Path(tmp).joinpath("charts/out").glob("*.png"))
            self.assertEqual(len(pngs), 1)
            self.assertTrue(pngs[0].name.endswith("-5m1h.png"))
            from PIL import Image

            with Image.open(pngs[0]) as im:
                self.assertGreater(im.height, 1000)
        self.assertIn("南亞科", text)
        self.assertIn("十五分K", text)
        self.assertIn("十五分K / MA5 10 20", text)
        self.assertIn("小時K", text)
        self.assertIn("上＝五分K", text)
        self.assertIn("均線發散", text)
        self.assertIn("成交額前 250", text)
        self.assertIn("股價 &lt; 500", text)
        self.assertIn("一小時後", text)
        self.assertIn("進場後一小時勝率", text)
        self.assertNotIn("小時 MA20 不下彎", text)
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

    def test_skips_when_the_session_has_no_full_hour(self) -> None:
        idx = _session_index([date(2026, 8, 21)])
        df = _ohlcv([100.0] * len(idx), idx)
        move = hour_later(df, idx[-5], 100.0)
        self.assertIsNone(move)

    def test_summary_counts_wins_and_short_sessions(self) -> None:
        df = _history_then_live([99.0, 105.0] + [106.0] * 20)
        hits = iter_5m_ma200_alerts(df)
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

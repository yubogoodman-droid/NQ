from __future__ import annotations

import unittest
from types import SimpleNamespace

import pandas as pd

from tw.shioaji_feed import (
    apply_tick,
    concat_daily_frames,
    kbars_to_frame,
    minute_of_tick,
    resample_ohlcv,
    yahoo_symbol_to_code,
    _ranked_from_snap,
    _sj_busy,
)
from tw.kline import set_kline_source, using_shioaji


class ShioajiFeedTests(unittest.TestCase):
    def test_yahoo_symbol_to_code(self) -> None:
        self.assertEqual(yahoo_symbol_to_code("2330.TW"), "2330")
        self.assertEqual(yahoo_symbol_to_code("6182.TWO"), "6182")
        self.assertEqual(yahoo_symbol_to_code("6147"), "6147")

    def test_kbars_to_frame(self) -> None:
        kbars = SimpleNamespace(
            dict=lambda: {
                "ts": [
                    pd.Timestamp("2026-08-17 09:01", tz="Asia/Taipei"),
                    pd.Timestamp("2026-08-17 09:02", tz="Asia/Taipei"),
                ],
                "Open": [100.0, 101.0],
                "High": [101.0, 102.0],
                "Low": [99.0, 100.5],
                "Close": [101.0, 101.5],
                "Volume": [10, 12],
            }
        )
        df = kbars_to_frame(kbars)
        self.assertEqual(list(df.columns), ["open", "high", "low", "close", "volume"])
        self.assertEqual(len(df), 2)
        self.assertEqual(df["close"].iloc[-1], 101.5)

    def test_resample_1m_to_5m(self) -> None:
        idx = pd.date_range("2026-08-17 09:00", periods=10, freq="1min", tz="Asia/Taipei")
        df = pd.DataFrame(
            {
                "open": [10.0] * 10,
                "high": [11.0] * 10,
                "low": [9.0] * 10,
                "close": list(range(10, 20)),
                "volume": [1.0] * 10,
            },
            index=idx,
        )
        five = resample_ohlcv(df, "5min")
        self.assertEqual(len(five), 2)
        self.assertEqual(five["open"].iloc[0], 10.0)
        self.assertEqual(five["close"].iloc[0], 14.0)
        self.assertEqual(five["volume"].iloc[0], 5.0)

    def test_concat_daily_frames_keeps_latest_dup(self) -> None:
        tz = "Asia/Taipei"
        a = pd.DataFrame(
            {"open": [1], "high": [2], "low": [1], "close": [1.5], "volume": [10]},
            index=[pd.Timestamp("2026-08-14 13:30", tz=tz)],
        )
        b = pd.DataFrame(
            {
                "open": [1, 3],
                "high": [2, 4],
                "low": [1, 3],
                "close": [9.0, 3.5],
                "volume": [10, 8],
            },
            index=[
                pd.Timestamp("2026-08-14 13:30", tz=tz),
                pd.Timestamp("2026-08-17 09:01", tz=tz),
            ],
        )
        merged = concat_daily_frames([a, b])
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged["close"].iloc[0], 9.0)
        self.assertEqual(merged["close"].iloc[-1], 3.5)

    def test_tick_rolls_to_next_minute(self) -> None:
        open_bars: dict = {}
        frames: dict = {}
        tz = "Asia/Taipei"
        apply_tick(
            open_bars,
            frames,
            code="2330.TW",
            price=100.0,
            volume=1,
            ts=pd.Timestamp("2026-08-17 09:00:08", tz=tz),
        )
        self.assertEqual(open_bars["2330.TW"]["ts"], pd.Timestamp("2026-08-17 09:01", tz=tz))
        closed = apply_tick(
            open_bars,
            frames,
            code="2330.TW",
            price=102.0,
            volume=2,
            ts=pd.Timestamp("2026-08-17 09:01:01", tz=tz),
        )
        self.assertEqual(closed, pd.Timestamp("2026-08-17 09:01", tz=tz))
        self.assertEqual(frames["2330.TW"]["close"].iloc[-1], 100.0)
        self.assertEqual(open_bars["2330.TW"]["close"], 102.0)

    def test_minute_of_tick_ceils(self) -> None:
        ts = pd.Timestamp("2026-08-17 09:10:00", tz="Asia/Taipei")
        self.assertEqual(minute_of_tick(ts), ts)

    def test_using_shioaji_false_without_keys(self) -> None:
        set_kline_source("yahoo")
        self.assertFalse(using_shioaji())
        set_kline_source("auto")
        self.assertFalse(using_shioaji())

    def test_snapshot_rank_maps_otc(self) -> None:
        from tw.ranking import RankedStock

        snap = SimpleNamespace(
            code="6147",
            close=168.5,
            total_amount=5.4e9,
            total_volume=32000,
            exchange="OTC",
            name="頎邦",
            change_price=1.0,
            change_rate=0.6,
        )
        stock = _ranked_from_snap(snap, RankedStock)
        self.assertIsNotNone(stock)
        assert stock is not None
        self.assertEqual(stock.symbol, "6147.TWO")
        self.assertEqual(stock.exchange, "TWO")
        self.assertEqual(stock.turnover, 5.4e9)

    def test_sj_busy_detects_exclusive_access(self) -> None:
        self.assertTrue(_sj_busy(RuntimeError("fetch_contracts: exclusive access lost")))
        self.assertTrue(_sj_busy(TimeoutError("kbars timeout")))
        self.assertFalse(_sj_busy(ValueError("bad contract")))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from tw.ranking import RankedStock
from tw.report import build_k_chart, save_scan_html
from tw.screener import ScanHit, ScanResult
from tw.signals import AlertSnapshot


TAIPEI = ZoneInfo("Asia/Taipei")


def _bars(n: int = 240) -> pd.DataFrame:
    idx = pd.date_range("2026-08-17 09:00", periods=n, freq="1min", tz=TAIPEI)
    close = pd.Series(100.0, index=idx)
    close.iloc[-1] = 110.0
    return pd.DataFrame(
        {
            "open": close.shift(1).fillna(100.0),
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 1000.0,
        }
    )


class ReportChartTests(unittest.TestCase):
    def test_k_chart_contains_candlestick(self) -> None:
        df = _bars()
        ts = df.index[-1]
        hit = ScanHit(
            stock=RankedStock(1, "2408.TW", "南亞科", 110.0, 10.0, 10.0, 1, 1e9, "TAI"),
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
            bars=len(df),
            frame=df,
        )
        html = build_k_chart(hit)
        self.assertIn('"type":"candlestick"', html)
        self.assertIn("MA200", html)
        self.assertIn("triangle-up", html)

    def test_save_html_embeds_plotly(self) -> None:
        df = _bars()
        ts = df.index[-1]
        hit = ScanHit(
            stock=RankedStock(6, "1303.TW", "南亞", 208.5, 1.0, 0.5, 1, 1e10, "TAI"),
            snapshot=AlertSnapshot(
                timestamp=ts,
                close=209.0,
                prev_close=207.5,
                ma5=207.8,
                ma10=207.6,
                ma20=207.5,
                ma200=207.5,
                prev_ma200=207.5,
            ),
            bars=len(df),
            frame=df,
        )
        result = ScanResult(
            scanned_at=datetime(2026, 8, 17, 11, 0, tzinfo=TAIPEI),
            rank_time="2026-08-17T11:00:00+08:00",
            universe=[],
            candidates=[],
            hits=[hit],
        )
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = save_scan_html(result, Path(tmp) / "index.html")
            text = path.read_text(encoding="utf-8")
        self.assertIn("plotly-2.35.2", text)
        self.assertIn('"type":"candlestick"', text)
        self.assertIn("南亞", text)


if __name__ == "__main__":
    unittest.main()

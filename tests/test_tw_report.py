from __future__ import annotations

import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from tw.ranking import RankedStock
from tw.report import CHART_BARS_1M, _axis_ylim, _centered_ylim, build_k_chart, save_scan_html, save_week_index
from tw.screener import ScanHit, ScanResult
from tw.signals import AlertSnapshot


TAIPEI = ZoneInfo("Asia/Taipei")


def _bars(n: int = 240, freq: str = "1min") -> pd.DataFrame:
    idx = pd.date_range("2026-08-17 09:00", periods=n, freq=freq, tz=TAIPEI)
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


def _hit(*, frame: pd.DataFrame, frame_5m: pd.DataFrame | None = None) -> ScanHit:
    ts = frame.index[-1]
    return ScanHit(
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
        bars=len(frame),
        frame=frame,
        frame_5m=frame_5m,
    )


class ReportChartTests(unittest.TestCase):
    def test_1m_chart_zooms_around_signal(self) -> None:
        self.assertEqual(CHART_BARS_1M, (24, 10))

    def test_centered_ylim_keeps_small_ma_gap_small(self) -> None:
        lo, hi = _centered_ylim(267.38, 268.0, span_pct=0.03)
        self.assertGreaterEqual(hi - lo, 268.0 * 0.03)
        self.assertLess(0.46 / (hi - lo), 0.08)
        self.assertLess(abs((lo + hi) / 2 - 267.38), 0.05)

    def test_axis_ylim_pads_tight_1m_range(self) -> None:
        idx = pd.date_range("2026-08-17 11:00", periods=10, freq="1min", tz=TAIPEI)
        window = pd.DataFrame(
            {
                "low": [265.0] * 10,
                "high": [268.5] * 10,
                "ma20": [267.15] * 10,
                "ma200": [267.61] * 10,
            },
            index=idx,
        )
        lo, hi = _axis_ylim(window, 268.0, min_span_pct=0.03)
        self.assertGreaterEqual(hi - lo, 268.0 * 0.03)
        # 0.46 元的 MA20/MA200 差不該佔滿縱軸
        self.assertLess(0.46 / (hi - lo), 0.12)

    def test_k_chart_contains_candlestick(self) -> None:
        html = build_k_chart(_hit(frame=_bars()))
        self.assertIn("data:image/png;base64,", html)
        png = html.split("base64,", 1)[1].split('"', 1)[0]
        self.assertGreater(len(png), 200)

    def test_5m_chart_uses_5m_frame(self) -> None:
        hit = _hit(frame=_bars(), frame_5m=_bars(n=80, freq="5min"))
        html = build_k_chart(hit, timeframe="5m")
        self.assertIn("data:image/png;base64,", html)
        self.assertIn("五分K", html)

    def test_5m_chart_missing_frame_shows_placeholder(self) -> None:
        html = build_k_chart(_hit(frame=_bars()), timeframe="5m")
        self.assertNotIn("data:image/png;base64,", html)
        self.assertIn("無五分 K 線資料", html)

    def test_save_html_embeds_1m_and_5m_png(self) -> None:
        hit = _hit(frame=_bars(), frame_5m=_bars(n=80, freq="5min"))
        result = ScanResult(
            scanned_at=datetime(2026, 8, 17, 11, 0, tzinfo=TAIPEI),
            rank_time="2026-08-17T11:00:00+08:00",
            universe=[],
            candidates=[],
            hits=[hit],
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = save_scan_html(result, Path(tmp) / "index.html")
            text = path.read_text(encoding="utf-8")
            self.assertTrue((path.parent / "charts" / "index" / "1303.TW-1m.png").exists())
            self.assertTrue((path.parent / "charts" / "index" / "1303.TW-5m.png").exists())
            md = path.with_suffix(".md").read_text(encoding="utf-8")
        self.assertIn("1303.TW-1m.png", text)
        self.assertIn("1303.TW-5m.png", text)
        self.assertNotIn("data:image/png;base64,", text)
        self.assertIn("南亞", md)
        self.assertIn("1303.TW-1m.png", md)
        self.assertIn("南亞", text)
        self.assertIn("一分 K", text)
        self.assertIn("五分 K（對照）", text)
        self.assertIn("五分收盤 &gt; MA200", text)
        self.assertIn("MA20 靠近 MA200", text)
        self.assertIn("MA5", text)
        self.assertIn("MA200", text)

    def test_week_index_lists_each_day(self) -> None:
        hit = _hit(frame=_bars(), frame_5m=_bars(n=80, freq="5min"))
        results = [
            ScanResult(
                scanned_at=datetime(2026, 8, 17, 11, 0, tzinfo=TAIPEI),
                rank_time="盤後",
                universe=[],
                candidates=[],
                hits=[hit] if day == date(2026, 8, 10) else [],
                as_of=day,
            )
            for day in (
                date(2026, 8, 10),
                date(2026, 8, 11),
                date(2026, 8, 12),
                date(2026, 8, 13),
                date(2026, 8, 14),
            )
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = save_week_index(results, Path(tmp) / "week-last.md")
            text = path.read_text(encoding="utf-8")
        self.assertIn("週一 2026-08-10", text)
        self.assertIn("週五 2026-08-14", text)
        self.assertIn("backtest-2026-08-10.md", text)
        self.assertIn("南亞", text)
        self.assertIn("| 1 |", text)


if __name__ == "__main__":
    unittest.main()

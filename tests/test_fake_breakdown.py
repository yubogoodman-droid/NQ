"""假跌破後上拉單元測試。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.run_fake_breakdown import make_sample_fake_breakdown_bars
from nq.spring import detect_fake_breakdowns
from nq.strategy import FakeBreakdownStrategy


def _bars(
    rows: list[tuple[float, float, float, float, float]],
    start: str = "2026-08-14 09:00",
) -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(rows), freq="1min")
    return pd.DataFrame(rows, index=idx, columns=["open", "high", "low", "close", "volume"])


def _flat(n: int, px: float, vol: float = 200, noise: float = 0.25) -> list[tuple[float, float, float, float, float]]:
    rows = []
    for i in range(n):
        o = px + (0.05 if i % 2 == 0 else -0.05)
        rows.append((o, px + noise, px - noise, px, vol))
    return rows


class FakeBreakdownTests(unittest.TestCase):
    def test_detects_jinju_like_spring(self) -> None:
        df = make_sample_fake_breakdown_bars()
        patterns = detect_fake_breakdowns(df)
        self.assertGreaterEqual(len(patterns), 1)
        p = patterns[0]
        self.assertLess(p.spring_low, p.support)
        self.assertGreater(p.resistance, p.support)
        self.assertGreater(p.break_pct, 0.005)
        self.assertLess(p.break_pct, 0.03)
        self.assertIsNotNone(p.breakout_idx)
        self.assertGreater(df["close"].iloc[p.breakout_idx], p.resistance)
        self.assertGreater(p.volume_ratio, 1.15)

        signals = FakeBreakdownStrategy().generate_signals(df)
        self.assertGreaterEqual(len(signals), 1)
        sig = signals[0]
        self.assertGreater(sig.entry, sig.stop_loss)
        self.assertGreater(sig.target, sig.entry)

    def test_catches_delayed_breakout_like_jinju(self) -> None:
        """8358 金居 2026-08-14：09:35 站回、09:53 才收盤站上箱頂（約 18 根）。"""
        rows = _flat(50, 420.0)
        rows.append((418.0, 418.5, 413.0, 414.0, 180))
        rows.append((414.0, 420.5, 413.5, 420.2, 190))
        for _ in range(17):
            rows.append((420.1, 420.2, 419.6, 420.0, 200))
        rows.append((420.2, 426.0, 420.0, 425.0, 520))
        df = _bars(rows)

        tight = detect_fake_breakdowns(df, max_breakout_bars=12)
        self.assertEqual(tight, [])

        patterns = detect_fake_breakdowns(df)
        self.assertGreaterEqual(len(patterns), 1)
        p = patterns[0]
        self.assertLessEqual(p.reclaim_idx + 12, p.breakout_idx)
        self.assertGreater(df["close"].iloc[p.breakout_idx], p.resistance)

    def test_neckline_ignores_upper_wick(self) -> None:
        """單根上影不該把頸線撐高；收盤過實體高即可進場。"""
        rows = _flat(55, 420.0, noise=0.3)
        rows[45] = (420.0, 425.0, 420.0, 421.0, 200)
        rows.append((418.0, 418.5, 413.0, 414.0, 180))
        rows.append((414.0, 420.5, 413.5, 420.2, 190))
        rows.append((420.5, 424.5, 420.0, 424.0, 520))
        df = _bars(rows)
        patterns = detect_fake_breakdowns(df)
        self.assertGreaterEqual(len(patterns), 1)
        p = patterns[0]
        self.assertLess(p.resistance, 425.0)
        self.assertLess(p.resistance, p.box_high + 1e-9)
        self.assertGreater(df["close"].iloc[p.breakout_idx], p.resistance)
        self.assertAlmostEqual(float(df["close"].iloc[p.breakout_idx]), 424.0)

    def test_ignores_breakdown_without_reclaim(self) -> None:
        rows = _flat(50, 420.0)
        px = 420.0
        for _ in range(12):
            px -= 2.0
            rows.append((px + 0.5, px + 0.8, px - 0.4, px, 180))
        df = _bars(rows)
        self.assertEqual(detect_fake_breakdowns(df), [])

    def test_ignores_too_deep_breakdown(self) -> None:
        rows = _flat(50, 420.0)
        # 跌超過 3%（約 12.6 點）再拉回，視為真跌破
        for px in (410.0, 405.0, 400.0, 406.0, 421.0, 425.0):
            vol = 500 if px >= 421 else 180
            rows.append((px + 1, px + 1.5, px - 0.5, px, vol))
        df = _bars(rows)
        self.assertEqual(detect_fake_breakdowns(df), [])

    def test_ignores_breakout_without_volume(self) -> None:
        df = make_sample_fake_breakdown_bars()
        # 把上拉段量能壓到盤整水準
        df.loc[df["close"] > 421, "volume"] = 180
        self.assertEqual(detect_fake_breakdowns(df, require_volume=True), [])
        self.assertGreaterEqual(len(detect_fake_breakdowns(df, require_volume=False)), 1)

    def test_ignores_wide_chop(self) -> None:
        rows = []
        for i in range(80):
            px = 420 + (12 if i % 2 == 0 else -12)
            rows.append((px, px + 2, px - 2, px, 400))
        df = _bars(rows)
        self.assertEqual(detect_fake_breakdowns(df), [])

    def test_rejects_overnight_box(self) -> None:
        df = make_sample_fake_breakdown_bars()
        idx = [t - pd.Timedelta(days=1) if i < 50 else t for i, t in enumerate(df.index)]
        shifted = df.copy()
        shifted.index = pd.DatetimeIndex(idx)
        overnight = detect_fake_breakdowns(shifted, same_session=False, skip_open_minutes=0)
        filtered = detect_fake_breakdowns(shifted, same_session=True, skip_open_minutes=0)
        for p in filtered:
            self.assertEqual(shifted.index[p.range_start_idx].date(), shifted.index[p.breakout_idx].date())
        self.assertTrue(
            len(filtered) < len(overnight)
            or all(
                shifted.index[p.range_start_idx].date() == shifted.index[p.breakout_idx].date()
                for p in overnight
            )
        )

    def test_skips_first_minutes(self) -> None:
        df = make_sample_fake_breakdown_bars()
        late = detect_fake_breakdowns(df, skip_open_minutes=5)
        self.assertGreaterEqual(len(late), 1)
        for p in late:
            t = df.index[p.breakout_idx]
            self.assertGreaterEqual(t.hour * 60 + t.minute, 9 * 60 + 5)


if __name__ == "__main__":
    unittest.main()

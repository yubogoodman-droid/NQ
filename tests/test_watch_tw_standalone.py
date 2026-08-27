from __future__ import annotations

import ast
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.test_tw_5m import _history_then_live
from tests.test_tw_live import TAIPEI
import pandas as pd

import watch_tw


ROOT = Path(__file__).resolve().parents[1]


class StandaloneWatcherTests(unittest.TestCase):
    def test_file_does_not_import_tw_package(self) -> None:
        tree = ast.parse((ROOT / "watch_tw.py").read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".", 1)[0] == "tw":
                self.fail(f"standalone file imports {node.module}")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "tw" or alias.name.startswith("tw."):
                        self.fail(f"standalone file imports {alias.name}")

    def test_same_5m_alert_as_package(self) -> None:
        df = _history_then_live([99.0, 105.0])
        hits = watch_tw.iter_5m_ma240_alerts(df)
        self.assertEqual(len(hits), 1)
        bar = watch_tw.OhlcvBar(pd.Timestamp("2026-08-21 09:05:00", tz=TAIPEI), 105, 105, 105, 105)
        self.assertEqual(len(watch_tw.alerts_on_closed_bar(df, bar, tf="5m")), 1)
        text = watch_tw.format_telegram("創見", "2451.TW", hits[0], "5m")
        self.assertIn("五分K 剛站上 MA240", text)

    def test_same_5m_short_alert_as_package(self) -> None:
        df = _history_then_live([101.0, 95.0])
        hits = watch_tw.iter_5m_ma240_alerts(df, side="short")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].side, "short")
        bar = watch_tw.OhlcvBar(pd.Timestamp("2026-08-21 09:05:00", tz=TAIPEI), 95, 95, 95, 95)
        self.assertEqual(len(watch_tw.alerts_on_closed_bar(df, bar, tf="5m", side="short")), 1)
        text = watch_tw.format_telegram("創見", "2451.TW", hits[0], "5m")
        self.assertIn("五分K 剛跌破 MA240", text)


    def test_default_scan_pool_is_turnover_top_200(self) -> None:
        self.assertEqual(watch_tw.TURNOVER_TOP, 200)
        with patch("sys.argv", ["watch_tw.py"]):
            args = watch_tw.parse_args()
        self.assertEqual(args.top, 200)


if __name__ == "__main__":
    unittest.main()

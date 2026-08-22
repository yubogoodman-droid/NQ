"""南亞科一分均線邏輯測試。"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from nq.ma_site import save_backtest_site
from nq.nanya_ma import NanyaMaStrategy, add_nanya_features, run_nanya_ma_backtest, summarize_ma_trades


def _series(closes: list[float], *, vol: float = 400) -> pd.DataFrame:
    n = len(closes)
    idx = pd.date_range("2026-08-21 07:00", periods=n, freq="1min", tz="Asia/Taipei")
    c = np.array(closes, dtype=float)
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) + 0.12
    l = np.minimum(o, c) - 0.12
    v = np.full(n, vol, dtype=float)
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v}, index=idx)


def _coil_then_lift() -> pd.DataFrame:
    closes = [400.0 + 0.25 * np.sin(i / 9) for i in range(280)]
    last = closes[-1]
    for i in range(24):
        last += 0.22
        closes.append(last)
    df = _series(closes)
    df.loc[df.index[280:300], "volume"] = 1600
    df.loc[df.index[:280], "high"] = np.minimum(df["high"].iloc[:280], 405.0)
    df.loc[df.index[:280], "low"] = np.maximum(df["low"].iloc[:280], 395.0)
    return df


class NanyaMaTests(unittest.TestCase):
    def test_coil_then_breakout_creates_signal(self) -> None:
        df = _coil_then_lift()
        strategy = NanyaMaStrategy(tick_size=0.05)
        signals = strategy.generate_signals(df)
        self.assertTrue(signals)
        sig = signals[0]
        self.assertGreater(sig.ma5, sig.ma10)
        self.assertGreater(sig.ma10, sig.ma20)
        self.assertLess(sig.ext_200_pct, 0.035)
        self.assertGreater(sig.entry, sig.stop_loss)

    def test_extended_stack_like_436_is_rejected(self) -> None:
        # 模擬已經噴完：價遠高於 MA200，短均大幅扇開
        closes = list(np.linspace(400, 450, 260))
        df = _series(closes, vol=900)
        work = add_nanya_features(df)
        last = work.iloc[-1]
        self.assertGreater(float(last["close"] / last["ma200"] - 1), 0.03)
        strategy = NanyaMaStrategy(tick_size=0.05, max_ext_200=0.035)
        signals = strategy.generate_signals(df)
        late = [s for s in signals if s.ext_200_pct > 0.035]
        self.assertEqual(late, [])

    def test_lost_ma20_exits(self) -> None:
        df = _coil_then_lift()
        # 進場後做成跌破 MA20
        strategy = NanyaMaStrategy(tick_size=0.05, max_bars_hold=25)
        signals = strategy.generate_signals(df)
        self.assertTrue(signals)
        entry = signals[0].bar_idx
        for i in range(entry + 2, min(entry + 10, len(df))):
            df.iloc[i, df.columns.get_loc("close")] = 396.0
            df.iloc[i, df.columns.get_loc("low")] = 395.5
            df.iloc[i, df.columns.get_loc("high")] = 397.0
            df.iloc[i, df.columns.get_loc("open")] = 397.0
        trades = run_nanya_ma_backtest(
            df, symbol="TEST", strategy=strategy, cost_bps=0, flatten_minutes=None
        )
        self.assertTrue(trades)
        self.assertIn(trades[0].exit_reason, {"lost_ma20", "stop_loss"})

    def test_backtest_site_contains_charts(self) -> None:
        df = _coil_then_lift()
        strategy = NanyaMaStrategy(tick_size=0.05)
        trades = run_nanya_ma_backtest(df, symbol="DEMO.2408", strategy=strategy, cost_bps=0, flatten_minutes=None)
        self.assertTrue(trades)
        out = Path("/tmp/nanya_ma_site_test.html")
        save_backtest_site(
            out,
            title="測試",
            trades=trades,
            frames={"DEMO.2408": df},
            notes=["測試"],
            symbol_stats=[("DEMO.2408", summarize_ma_trades(trades), len(df))],
        )
        text = out.read_text(encoding="utf-8")
        self.assertIn("data:image/png;base64,", text)
        self.assertIn("<img", text)
        self.assertIn("DEMO.2408", text)
        self.assertIn("MA5", text)


if __name__ == "__main__":
    unittest.main()

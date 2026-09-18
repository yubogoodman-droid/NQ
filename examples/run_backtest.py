#!/usr/bin/env python3
"""NQ 五分 K 破三小時低 5/20 多排站上 MA60 範例。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nq.backtest import run_backtest, summarize
from nq.strategy import NQWBottomStrategy


def make_sample_bars(n: int = 80, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = 18000.0
    rows = []
    px = base
    for i in range(n):
        if i == 40:
            o, h, l, c = px, px + 4, px - 50, px - 10
        elif i == 41:
            o, h, l, c = px - 10, px + 20, px - 12, px + 18
        else:
            o = px
            c = px + float(rng.normal(0, 1.0))
            h = max(o, c) + 3
            l = min(o, c) - 3
            px = c
        rows.append((o, h, l, c))
        px = c
    idx = pd.date_range("2026-08-07 09:30", periods=n, freq="5min")
    return pd.DataFrame(
        [{"open": o, "high": max(o, h, c), "low": min(o, l, c), "close": c, "volume": 200} for o, h, l, c in rows],
        index=idx,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="NQ 五分 K 破三小時低 5/20站上MA60")
    parser.add_argument("--csv", help="五分 K CSV 路徑（需含 datetime,open,high,low,close）")
    parser.add_argument("--demo", action="store_true", help="使用模擬資料示範")
    args = parser.parse_args()

    if args.csv:
        df = pd.read_csv(args.csv, parse_dates=["datetime"], index_col="datetime")
    else:
        df = make_sample_bars()
        print("使用模擬破底翻資料\n")

    strategy = NQWBottomStrategy()
    signals = strategy.generate_signals(df)
    print("=== 破三小時低 5/20站上MA60 ===")
    if not signals:
        print("未偵測到進場訊號")
    for sig in signals:
        print(f"{sig.timestamp} | 做多 @ {sig.entry:.2f} | 停損 {sig.stop_loss:.2f} | 目標 {sig.target:.2f}")
    results = run_backtest(df, strategy)
    print(summarize(results))


if __name__ == "__main__":
    main()

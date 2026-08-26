#!/usr/bin/env python3
"""NQ 五分 K W 底策略範例。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nq.backtest import run_backtest, summarize
from nq.strategy import NQWBottomStrategy


def make_sample_w_bottom_bars(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """產生含破底 W 底型態的模擬五分 K 資料。"""
    rng = np.random.default_rng(seed)
    base = 18000.0
    prices = [base]

    for i in range(1, n):
        drift = 0.0
        if 40 <= i <= 52:
            drift = -3.0
        elif 53 <= i <= 68:
            drift = 4.2
        elif 69 <= i <= 74:
            drift = -8.0  # 破底
        elif 75 <= i <= 88:
            drift = 3.5
        elif 89 <= i <= 96:
            drift = -2.2  # L2 回測
        elif i > 96:
            drift = 3.0
        prices.append(prices[-1] + drift + rng.normal(0, 1.2))

    closes = np.array(prices)
    highs = closes + rng.uniform(0.5, 3.0, n)
    lows = closes - rng.uniform(0.5, 3.0, n)
    # 強制中間破底長影線，讓偵測穩定
    lows[72] = min(lows[72], closes[52] - 18.0)
    opens = np.roll(closes, 1)
    opens[0] = base

    idx = pd.date_range("2026-08-07 09:30", periods=n, freq="5min")
    return pd.DataFrame(
        {"open": opens, "high": np.maximum(highs, np.maximum(opens, closes)),
         "low": np.minimum(lows, np.minimum(opens, closes)),
         "close": closes, "volume": rng.integers(100, 1000, n)},
        index=idx,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="NQ 五分 K W 底進場策略")
    parser.add_argument("--csv", help="五分 K CSV 路徑（需含 datetime,open,high,low,close）")
    parser.add_argument("--demo", action="store_true", help="使用模擬資料示範")
    args = parser.parse_args()

    if args.csv:
        df = pd.read_csv(args.csv, parse_dates=["datetime"], index_col="datetime")
    elif args.demo or not args.csv:
        df = make_sample_w_bottom_bars()
        print("使用模擬 W 底資料\n")
    else:
        parser.error("請提供 --csv 或使用 --demo")

    strategy = NQWBottomStrategy()
    signals = strategy.generate_signals(df)

    print("=== W 底進場訊號 ===")
    if not signals:
        print("未偵測到 W 底 L3 收盤進場訊號")
    for sig in signals:
        print(
            f"{sig.timestamp} | 做多 @ {sig.entry:.2f} | "
            f"停損 {sig.stop_loss:.2f} | 目標 {sig.target:.2f} | "
            f"風險 {sig.risk:.2f} 點"
        )

    results = run_backtest(df, strategy)
    stats = summarize(results)
    print("\n=== 回測摘要 ===")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"{k}: {v:.2f}")
        else:
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""假跌破後上拉策略範例（1 分 K）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nq.backtest import run_backtest, summarize
from nq.spring import FakeBreakdownPattern
from nq.strategy import FakeBreakdownStrategy


def make_sample_fake_breakdown_bars(seed: int = 7) -> pd.DataFrame:
    """
    產生類似 8358 金居 1 分 K 的假跌破後噴出：
    先在 419–421 盤整，假跌到 413，站回後放量上拉。
    """
    rng = np.random.default_rng(seed)
    rows: list[tuple[float, float, float, float, int]] = []

    def add_bar(close: float, *, high_off: float, low_off: float, volume: int, open_off: float = 0.0) -> None:
        o = close + open_off
        h = max(o, close) + high_off
        l = min(o, close) - low_off
        rows.append((o, h, l, close, volume))

    price = 420.5
    for _ in range(40):
        price = 420.4 + float(rng.normal(0, 0.12))
        add_bar(price, high_off=0.4, low_off=0.4, volume=int(rng.integers(180, 240)), open_off=float(rng.normal(0, 0.15)))

    for _ in range(18):
        price = 420.0 + float(rng.normal(0, 0.08))
        price = min(max(price, 419.4), 420.6)
        add_bar(price, high_off=0.35, low_off=0.35, volume=int(rng.integers(160, 220)), open_off=float(rng.normal(0, 0.1)))

    spring = [418.6, 417.2, 415.4, 413.2, 413.0, 414.8, 417.5]
    for i, px in enumerate(spring):
        add_bar(
            px,
            high_off=0.5 if i < 4 else 1.0,
            low_off=0.4,
            volume=int(rng.integers(140, 200)),
            open_off=0.8 if i < 4 else -0.6,
        )

    # 站回後先在箱內磨，約 18 根才放量站上箱頂（對齊 8358 金居 09:35→09:53）
    add_bar(419.6, high_off=0.5, low_off=0.3, volume=int(rng.integers(200, 260)), open_off=-0.4)
    for _ in range(16):
        price = min(max(420.2 + float(rng.normal(0, 0.06)), 419.7), 420.55)
        add_bar(
            price,
            high_off=0.2,
            low_off=0.25,
            volume=int(rng.integers(160, 220)),
            open_off=float(rng.normal(0, 0.08)),
        )
    rally = [424.0, 428.5, 435.0, 444.0, 452.0, 459.0]
    for i, px in enumerate(rally):
        vol = int(rng.integers(380, 620))
        add_bar(px, high_off=1.2, low_off=0.4, volume=vol, open_off=-0.8)

    idx = pd.date_range("2026-08-14 09:00", periods=len(rows), freq="1min")
    return pd.DataFrame(
        rows,
        index=idx,
        columns=["open", "high", "low", "close", "volume"],
    )


def _print_pattern(df: pd.DataFrame, pattern: FakeBreakdownPattern) -> None:
    print(
        f"盤整 {df.index[pattern.range_start_idx].strftime('%H:%M')}–"
        f"{df.index[pattern.range_end_idx].strftime('%H:%M')} | "
        f"箱 {pattern.support:.2f}–{pattern.resistance:.2f} ({pattern.range_pct * 100:.2f}%)"
    )
    print(
        f"假跌破 {df.index[pattern.spring_idx].strftime('%H:%M')} @ {pattern.spring_low:.2f} "
        f"(跌破 {pattern.break_pct * 100:.2f}%)"
    )
    print(
        f"站回 {df.index[pattern.reclaim_idx].strftime('%H:%M')} | "
        f"放量比 {pattern.volume_ratio:.2f}x"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="假跌破後上拉進場策略")
    parser.add_argument("--csv", help="1 分 K CSV（需含 datetime,open,high,low,close,volume）")
    parser.add_argument("--demo", action="store_true", help="使用模擬金居型 1 分 K")
    parser.add_argument("--no-volume", action="store_true", help="不要求放量確認")
    parser.add_argument("--chart", help="輸出 HTML 圖表路徑")
    args = parser.parse_args()

    if args.csv:
        df = pd.read_csv(args.csv, parse_dates=["datetime"], index_col="datetime")
    else:
        df = make_sample_fake_breakdown_bars()
        print("使用模擬假跌破資料（金居 1 分 K 結構）\n")

    strategy = FakeBreakdownStrategy(require_volume=not args.no_volume)
    signals = strategy.generate_signals(df)

    print("=== 假跌破後上拉訊號 ===")
    if not signals:
        print("未偵測到符合條件的假跌破上拉")
    for sig in signals:
        assert isinstance(sig.pattern, FakeBreakdownPattern)
        print(
            f"{sig.timestamp} | 做多 @ {sig.entry:.2f} | "
            f"停損 {sig.stop_loss:.2f} | 目標 {sig.target:.2f} | "
            f"風險 {sig.risk:.2f}"
        )
        _print_pattern(df, sig.pattern)

    results = run_backtest(df, strategy, max_bars_hold=60)
    stats = summarize(results)
    print("\n=== 回測摘要 ===")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"{k}: {v:.2f}")
        else:
            print(f"{k}: {v}")

    if args.chart:
        from nq.spring_chart import save_spring_html_chart

        out = save_spring_html_chart(df, args.chart, strategy=strategy)
        print(f"\n已產生圖表: {out.resolve()}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""一分 K 棒型態回測：教科書 K 形 + 南亞科式盤整放量突破。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nq.candles import CORE_PATTERNS
from nq.one_min import (
    OneMinCandleStrategy,
    run_one_min_backtest,
    save_one_min_report,
    summarize_by_pattern,
    summarize_trades,
)

DEFAULT_SYMBOLS = ("2408.TW", "2344.TW", "2303.TW", "2330.TW", "NQ=F")


def tw_tick_size(price: float) -> float:
    if price < 10:
        return 0.01
    if price < 50:
        return 0.05
    if price < 100:
        return 0.1
    if price < 500:
        return 0.5
    if price < 1000:
        return 1.0
    return 5.0


def fetch_1m(symbol: str, period: str = "7d") -> pd.DataFrame:
    import yfinance as yf

    raw = yf.Ticker(symbol).history(period=period, interval="1m", auto_adjust=False)
    if raw.empty:
        raise RuntimeError(f"無法取得 {symbol} 一分 K")
    df = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].copy()
    df = df.dropna()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    if symbol.endswith(".TW") or symbol.endswith(".TWO"):
        df.index = df.index.tz_convert("Asia/Taipei")
        minutes = df.index.hour * 60 + df.index.minute
        df = df[(minutes >= 9 * 60) & (minutes <= 13 * 60 + 30)]
    elif symbol.endswith("=F"):
        df.index = df.index.tz_convert("America/New_York")
    return df


def make_demo_1m_bars(n: int = 420, seed: int = 7) -> pd.DataFrame:
    """模擬南亞科那種長盤整後放量長紅，並插入錘子／吞噬。"""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-08-21 09:00", periods=n, freq="1min", tz="Asia/Taipei")
    close = np.empty(n)
    close[0] = 400.0
    for i in range(1, n):
        if i < 80:
            close[i] = close[i - 1] + rng.normal(-0.08, 0.18)
        elif i < 280:
            close[i] = 395 + rng.normal(0, 0.35) + 0.08 * np.sin(i / 9)
        elif i < 300:
            close[i] = close[i - 1] + rng.normal(1.1, 0.25)
        else:
            close[i] = close[i - 1] + rng.normal(0.05, 0.35)

    # 錘子：盤整中段做一個長下影
    close[140] = 395.2
    low = close - rng.uniform(0.15, 0.45, n)
    high = close + rng.uniform(0.15, 0.45, n)
    open_ = np.r_[close[0], close[:-1]] + rng.normal(0, 0.08, n)
    low[140] = 393.1
    high[140] = 395.35
    open_[140] = 395.05
    close[140] = 395.25

    # 多頭吞噬
    open_[210] = 396.4
    close[210] = 394.9
    high[210] = 396.5
    low[210] = 394.8
    open_[211] = 394.7
    close[211] = 396.6
    high[211] = 396.7
    low[211] = 394.6

    # 突破段加大實體與量
    volume = rng.integers(400, 900, n).astype(float)
    volume[80:280] *= 0.55
    volume[280:295] *= 4.2
    high[280:300] = np.maximum(high[280:300], close[280:300] + 0.2)
    low[280:300] = np.minimum(low[280:300], open_[280:300] - 0.15)
    open_[280:300] = np.minimum(open_[280:300], close[280:300] - 0.8)

    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def _print_stats(title: str, trades, bars: int) -> None:
    stats = summarize_trades(trades)
    print(f"\n=== {title} | {bars} 根一分 K ===")
    print(
        f"成交 {stats['trades']}  勝率 {stats['win_rate']*100:.0f}%  "
        f"毛利 {stats['total_pnl_pct']*100:+.2f}%  淨利 {stats['total_pnl_pct_net']*100:+.2f}%  "
        f"期望 {stats['expectancy_net']*100:+.3f}%"
    )
    for row in summarize_by_pattern(trades):
        print(
            f"  {row['name_zh']:8s} {row['name']:22s}  "
            f"n={row['trades']:3d}  WR {row['win_rate']*100:5.1f}%  "
            f"淨 {row['total_pnl_pct_net']*100:+6.2f}%  "
            f"E {row['expectancy_net']*100:+6.3f}%"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="一分 K 棒型態回測")
    parser.add_argument("--csv", help="一分 K CSV（datetime,open,high,low,close,volume）")
    parser.add_argument("--demo", action="store_true", help="只用模擬南亞科式盤整突破")
    parser.add_argument("--symbols", nargs="*", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--period", default="7d", help="yfinance 週期，一分 K 最多 7d")
    parser.add_argument("--output", "-o", default="docs/1m-candles/index.html")
    parser.add_argument("--cost-bps", type=float, default=8.0, help="單邊成本 bps")
    args = parser.parse_args()

    all_trades = []
    symbol_stats: list[tuple[str, dict, int]] = []
    notes = [
        "進場：型態收盤確認後的下一根開盤，避免偷看當根高低。",
        "停損在型態低／高再加 0.15 ATR；停利 1.5R；超過 20 分鐘時間停。",
        "當日已漲／跌超過 6% 不再追（對應南亞科衝到 436 那種末端）。",
        "盤整放量突破：約 30–60 根區間寬度 ≤4%，當根放量長紅收在區間上。",
        "台股 13:20 以後強制平倉；成本預設單邊 8bps（往返 16bps）。",
        "Yahoo 一分 K 最多約 7 個交易日，樣本有限，結果只供學習。",
        "鑷子／錘子／光頭光腳在一分 K 出現太頻繁，核心統計不含這三類。",
    ]

    if args.csv:
        df = pd.read_csv(args.csv, parse_dates=["datetime"], index_col="datetime")
        last = float(df["close"].iloc[-1])
        strategy = OneMinCandleStrategy(tick_size=tw_tick_size(last))
        trades = run_one_min_backtest(df, symbol=Path(args.csv).name, strategy=strategy, cost_bps=args.cost_bps)
        all_trades.extend(trades)
        symbol_stats.append((Path(args.csv).name, summarize_trades(trades), len(df)))
        _print_stats(Path(args.csv).name, trades, len(df))
    elif args.demo:
        df = make_demo_1m_bars()
        strategy = OneMinCandleStrategy(tick_size=0.05)
        trades = run_one_min_backtest(
            df, symbol="DEMO.2408", strategy=strategy, cost_bps=args.cost_bps, flatten_minutes=(13, 20)
        )
        all_trades.extend(trades)
        symbol_stats.append(("DEMO.2408", summarize_trades(trades), len(df)))
        _print_stats("模擬南亞科盤整突破", trades, len(df))
    else:
        for symbol in args.symbols:
            try:
                df = fetch_1m(symbol, period=args.period)
            except Exception as exc:
                print(f"略過 {symbol}: {exc}")
                continue
            if len(df) < 80:
                print(f"略過 {symbol}: 只有 {len(df)} 根")
                continue
            last = float(df["close"].iloc[-1])
            tick = 0.25 if symbol.endswith("=F") else tw_tick_size(last)
            cost = 1.0 if symbol.endswith("=F") else args.cost_bps
            flatten = (13, 20) if symbol.endswith((".TW", ".TWO")) else None
            strategy = OneMinCandleStrategy(tick_size=tick)
            trades = run_one_min_backtest(
                df, symbol=symbol, strategy=strategy, cost_bps=cost, flatten_minutes=flatten
            )
            all_trades.extend(trades)
            symbol_stats.append((symbol, summarize_trades(trades), len(df)))
            _print_stats(symbol, trades, len(df))

    overall = summarize_trades(all_trades)
    core_trades = [t for t in all_trades if t.signal.pattern.name in CORE_PATTERNS]
    core = summarize_trades(core_trades)
    print("\n=== 全部合計 ===")
    print(
        f"成交 {overall['trades']}  勝率 {overall['win_rate']*100:.0f}%  "
        f"淨利 {overall['total_pnl_pct_net']*100:+.2f}%  期望 {overall['expectancy_net']*100:+.3f}%"
    )
    print("\n=== 核心型態（突破／吞噬／星線／三兵） ===")
    print(
        f"成交 {core['trades']}  勝率 {core['win_rate']*100:.0f}%  "
        f"淨利 {core['total_pnl_pct_net']*100:+.2f}%  期望 {core['expectancy_net']*100:+.3f}%"
    )
    for row in summarize_by_pattern(core_trades):
        print(
            f"  {row['name_zh']:8s} {row['name']:22s}  "
            f"n={row['trades']:3d}  WR {row['win_rate']*100:5.1f}%  "
            f"淨 {row['total_pnl_pct_net']*100:+6.2f}%  "
            f"E {row['expectancy_net']*100:+6.3f}%"
        )

    out = save_one_min_report(
        args.output,
        title="一分 K 棒型態回測",
        trades=all_trades,
        notes=notes,
        symbol_stats=symbol_stats,
    )
    print(f"\n報告：{out.resolve()}")


if __name__ == "__main__":
    main()

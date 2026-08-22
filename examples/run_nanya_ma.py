#!/usr/bin/env python3
"""南亞科一分圖均線回測：MA5/10/20/60/120/200。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nq.nanya_ma import NanyaMaStrategy, run_nanya_ma_backtest, save_nanya_ma_report, summarize_ma_trades

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


def make_nanya_ma_demo(n: int = 360, seed: int = 3) -> pd.DataFrame:
    """先在 395–405 黏均，再放量站上短均（對應截圖啟動段，不是 436）。"""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-08-21 08:00", periods=n, freq="1min", tz="Asia/Taipei")
    close = np.empty(n)
    close[0] = 400.0
    for i in range(1, n):
        if i < 280:
            close[i] = 400.0 + rng.normal(0, 0.28) + 0.35 * np.sin(i / 11)
        elif i < 300:
            close[i] = close[i - 1] + rng.normal(0.55, 0.12)
        else:
            close[i] = close[i - 1] + rng.normal(0.04, 0.22)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + rng.uniform(0.08, 0.22, n)
    low = np.minimum(open_, close) - rng.uniform(0.08, 0.22, n)
    high[:280] = np.clip(high[:280], None, 405.0)
    low[:280] = np.clip(low[:280], 395.0, None)
    volume = rng.integers(300, 700, n).astype(float)
    volume[:280] *= 0.6
    volume[280:295] *= 3.8
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def _print_stats(title: str, trades, bars: int) -> None:
    s = summarize_ma_trades(trades)
    print(f"\n=== {title} | {bars} 根一分 K ===")
    print(
        f"成交 {s['trades']}  勝率 {s['win_rate']*100:.0f}%  "
        f"毛利 {s['total_pnl_pct']*100:+.2f}%  淨利 {s['total_pnl_pct_net']*100:+.2f}%  "
        f"期望 {s['expectancy_net']*100:+.3f}%"
    )
    for t in trades[:8]:
        sig = t.signal
        print(
            f"  {_short_ts(sig.timestamp)} 進 {sig.entry:.2f}  "
            f"5/10/20 {sig.ma5:.1f}/{sig.ma10:.1f}/{sig.ma20:.1f}  "
            f"離200 {sig.ext_200_pct*100:.2f}%  5-20 {sig.short_span_pct*100:.2f}%  "
            f"→ {t.exit_reason} {t.pnl_pct_net*100:+.2f}%"
        )


def _short_ts(ts: pd.Timestamp) -> str:
    t = ts.tz_localize(None) if getattr(ts, "tzinfo", None) else ts
    return t.strftime("%m-%d %H:%M")


def main() -> None:
    parser = argparse.ArgumentParser(description="南亞科一分均線回測")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--symbols", nargs="*", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--period", default="7d")
    parser.add_argument("--output", "-o", default="docs/nanya-ma/index.html")
    parser.add_argument("--cost-bps", type=float, default=8.0)
    args = parser.parse_args()

    notes = [
        "均線：MA5/10/20/60/120/200，與南亞科一分截圖相同。",
        "盤整：近 30–60 根區間 ≤4%，突破前短三均中位寬度 ≤0.6%。",
        "進場：收盤站上盤整高與 MA20，且 MA5>MA10>MA20，5–20 扇開 0.15%–0.80%。",
        "價離 MA200 只要 0.20%–3.5%；已排成 5>…>200 且 5–200 超過 4% 視為 436 末端，不追。",
        "當日漲幅 >6%、09:10 前不進。下一根開盤成交。",
        "停損在 min(MA20, 區間低)；停利 1.5R；連續兩根收盤跌破 MA20 離場；最多 30 分鐘。",
        "台股 13:20 平倉。Yahoo 一分 K 約 7 日，只供學習。",
    ]

    all_trades = []
    symbol_stats: list[tuple[str, dict, int]] = []

    if args.demo:
        df = make_nanya_ma_demo()
        strategy = NanyaMaStrategy(tick_size=0.05)
        trades = run_nanya_ma_backtest(df, symbol="DEMO.2408", strategy=strategy, cost_bps=args.cost_bps)
        all_trades.extend(trades)
        symbol_stats.append(("DEMO.2408", summarize_ma_trades(trades), len(df)))
        _print_stats("模擬南亞科黏均後啟動", trades, len(df))
    else:
        for symbol in args.symbols:
            try:
                df = fetch_1m(symbol, period=args.period)
            except Exception as exc:
                print(f"略過 {symbol}: {exc}")
                continue
            if len(df) < 240:
                print(f"略過 {symbol}: 只有 {len(df)} 根")
                continue
            last = float(df["close"].iloc[-1])
            tick = 0.25 if symbol.endswith("=F") else tw_tick_size(last)
            cost = 1.0 if symbol.endswith("=F") else args.cost_bps
            flatten = None if symbol.endswith("=F") else (13, 20)
            if symbol.endswith("=F"):
                strategy = NanyaMaStrategy(tick_size=tick, session_open_hour=None)
            else:
                strategy = NanyaMaStrategy(tick_size=tick, entry_after_minute=10)
            trades = run_nanya_ma_backtest(
                df, symbol=symbol, strategy=strategy, cost_bps=cost, flatten_minutes=flatten
            )
            all_trades.extend(trades)
            symbol_stats.append((symbol, summarize_ma_trades(trades), len(df)))
            _print_stats(symbol, trades, len(df))

    overall = summarize_ma_trades(all_trades)
    print("\n=== 全部合計 ===")
    print(
        f"成交 {overall['trades']}  勝率 {overall['win_rate']*100:.0f}%  "
        f"淨利 {overall['total_pnl_pct_net']*100:+.2f}%  期望 {overall['expectancy_net']*100:+.3f}%"
    )
    out = save_nanya_ma_report(
        args.output,
        title="南亞科一分均線回測",
        trades=all_trades,
        notes=notes,
        symbol_stats=symbol_stats,
    )
    print(f"\n報告：{out.resolve()}")


if __name__ == "__main__":
    main()

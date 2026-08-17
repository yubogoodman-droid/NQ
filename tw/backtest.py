"""台股做空回測：跌破 MA200 當日收盤進場，站回 MA200 或到期出場。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tw.data import to_panels
from tw.strategy import TwMaShortStrategy, TwSignal
from tw.universe import TwStock, weekly_top_n_mask


@dataclass
class TradeResult:
    signal: TwSignal
    exit_price: float
    exit_time: pd.Timestamp
    exit_reason: str
    hold_days: int
    pnl_pct: float
    pnl_twd: float


def _round_trip_cost(commission: float, tax: float) -> float:
    # 放空：賣出抽證交稅 + 來回手續費
    return commission * 2 + tax


def run_symbol_backtest(
    df: pd.DataFrame,
    signals: list[TwSignal],
    *,
    max_hold_days: int = 20,
    stop_loss_pct: float = 0.08,
    take_profit_pct: float = 0.12,
    commission: float = 0.001425,
    tax: float = 0.003,
    notional: float = 100_000.0,
) -> list[TradeResult]:
    """
    出場（隔日不再用當根收盤同時進出場）：
    1. 收盤站回 MA200 → 以該收盤回補
    2. 反彈達停損（相對進場價）
    3. 下跌達停利
    4. 持有滿 max_hold_days
    """
    if not signals or df.empty:
        return []

    close = df["close"]
    high = df["high"]
    ma200 = close.rolling(200, min_periods=200).mean()
    cost = _round_trip_cost(commission, tax)
    results: list[TradeResult] = []
    busy_until = -1

    for sig in sorted(signals, key=lambda s: s.bar_idx):
        entry_idx = sig.bar_idx
        if entry_idx <= busy_until or entry_idx >= len(df) - 1:
            continue

        end_idx = min(entry_idx + max_hold_days, len(df) - 1)
        exit_idx = end_idx
        exit_reason = "time_stop"
        exit_price = float(close.iloc[end_idx])

        stop_price = sig.entry * (1 + stop_loss_pct)
        target_price = sig.entry * (1 - take_profit_pct)

        for i in range(entry_idx + 1, end_idx + 1):
            bar_high = float(high.iloc[i])
            bar_close = float(close.iloc[i])
            ma = ma200.iloc[i]

            if bar_high >= stop_price:
                exit_price = stop_price
                exit_idx = i
                exit_reason = "stop_loss"
                break
            if bar_close <= target_price:
                exit_price = target_price
                exit_idx = i
                exit_reason = "take_profit"
                break
            if pd.notna(ma) and bar_close >= float(ma):
                exit_price = bar_close
                exit_idx = i
                exit_reason = "ma200_reclaim"
                break

        pnl_pct = (sig.entry - exit_price) / sig.entry - cost
        results.append(
            TradeResult(
                signal=sig,
                exit_price=exit_price,
                exit_time=pd.Timestamp(df.index[exit_idx]),
                exit_reason=exit_reason,
                hold_days=int(exit_idx - entry_idx),
                pnl_pct=float(pnl_pct),
                pnl_twd=float(notional * pnl_pct),
            )
        )
        busy_until = exit_idx

    return results


def run_backtest(
    frames: dict[str, pd.DataFrame],
    stocks: list[TwStock],
    *,
    strategy: TwMaShortStrategy | None = None,
    top_n: int = 100,
    max_price: float = 600.0,
    max_hold_days: int = 20,
    stop_loss_pct: float = 0.08,
    take_profit_pct: float = 0.12,
    commission: float = 0.001425,
    tax: float = 0.003,
    notional: float = 100_000.0,
    start: str | None = None,
) -> list[TradeResult]:
    strategy = strategy or TwMaShortStrategy(max_price=max_price)
    names = {s.ticker: s.name for s in stocks}
    _opens, _highs, _lows, closes, volumes = to_panels(frames)
    eligible = weekly_top_n_mask(closes, volumes, top_n=top_n, max_price=max_price)

    results: list[TradeResult] = []
    for ticker, df in frames.items():
        if start:
            df = df[df.index >= pd.Timestamp(start)]
        if df.empty:
            continue
        elig = eligible[ticker] if ticker in eligible.columns else None
        if elig is not None and start:
            elig = elig[elig.index >= pd.Timestamp(start)]
        signals = strategy.generate_signals(
            df,
            ticker=ticker,
            name=names.get(ticker, ticker),
            eligible=elig,
        )
        results.extend(
            run_symbol_backtest(
                df,
                signals,
                max_hold_days=max_hold_days,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
                commission=commission,
                tax=tax,
                notional=notional,
            )
        )

    results.sort(key=lambda r: (r.signal.timestamp, r.signal.ticker))
    return results


def summarize(results: list[TradeResult]) -> dict:
    if not results:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "avg_pnl_pct": 0.0,
            "median_pnl_pct": 0.0,
            "total_pnl_twd": 0.0,
            "profit_factor": 0.0,
            "avg_hold_days": 0.0,
            "max_drawdown_twd": 0.0,
        }

    pnls = np.array([r.pnl_pct for r in results], dtype=float)
    twd = np.array([r.pnl_twd for r in results], dtype=float)
    wins = pnls > 0
    gross_win = float(twd[twd > 0].sum())
    gross_loss = float(-twd[twd < 0].sum())
    equity = np.cumsum(twd)
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    return {
        "trades": len(results),
        "wins": int(wins.sum()),
        "losses": int((~wins).sum()),
        "win_rate": float(wins.mean()),
        "avg_pnl_pct": float(pnls.mean()),
        "median_pnl_pct": float(np.median(pnls)),
        "total_pnl_twd": float(twd.sum()),
        "profit_factor": (gross_win / gross_loss) if gross_loss else float("inf"),
        "avg_hold_days": float(np.mean([r.hold_days for r in results])),
        "max_drawdown_twd": float(dd.min()) if len(dd) else 0.0,
    }

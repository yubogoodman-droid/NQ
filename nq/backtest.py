"""簡易回測引擎。"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from nq.strategy import NQWBottomStrategy, Signal


@dataclass
class TradeResult:
    signal: Signal
    exit_price: float
    exit_time: pd.Timestamp
    exit_reason: str
    pnl_points: float
    pnl_dollars: float


def run_backtest(
    df: pd.DataFrame,
    strategy: NQWBottomStrategy | None = None,
    max_bars_hold: int = 48,
) -> list[TradeResult]:
    """
    對歷史五分 K 執行 W 底策略回測。

    出場規則（依序檢查）：
    1. 觸及停損
    2. 觸及停利
    3. 持倉超過 max_bars_hold 根 K 強制平倉
    """
    strategy = strategy or NQWBottomStrategy()
    signals = strategy.generate_signals(df)
    results: list[TradeResult] = []

    for sig in signals:
        entry_idx = sig.bar_idx
        end_idx = min(entry_idx + max_bars_hold, len(df) - 1)

        exit_price = df["close"].iloc[end_idx]
        exit_time = df.index[end_idx]
        exit_reason = "time_stop"

        for i in range(entry_idx + 1, end_idx + 1):
            low = df["low"].iloc[i]
            high = df["high"].iloc[i]

            if low <= sig.stop_loss:
                exit_price = sig.stop_loss
                exit_time = df.index[i]
                exit_reason = "stop_loss"
                break
            if high >= sig.target:
                exit_price = sig.target
                exit_time = df.index[i]
                exit_reason = "take_profit"
                break

        pnl_points = exit_price - sig.entry
        results.append(
            TradeResult(
                signal=sig,
                exit_price=exit_price,
                exit_time=exit_time,
                exit_reason=exit_reason,
                pnl_points=pnl_points,
                pnl_dollars=pnl_points * strategy.point_value,
            )
        )

    return results


def summarize(results: list[TradeResult]) -> dict:
    if not results:
        return {"trades": 0, "win_rate": 0.0, "total_pnl": 0.0}

    wins = sum(1 for r in results if r.pnl_points > 0)
    return {
        "trades": len(results),
        "wins": wins,
        "losses": len(results) - wins,
        "win_rate": wins / len(results),
        "total_pnl_points": sum(r.pnl_points for r in results),
        "total_pnl_dollars": sum(r.pnl_dollars for r in results),
        "avg_pnl_points": sum(r.pnl_points for r in results) / len(results),
    }

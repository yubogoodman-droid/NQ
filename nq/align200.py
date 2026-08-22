"""一分 K：MA5>10>20>60 多頭排列，且收盤站上 MA200 才發訊。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

MA_PERIODS = (5, 10, 20, 60, 200)


@dataclass(frozen=True)
class Align200Signal:
    timestamp: pd.Timestamp
    bar_idx: int
    symbol: str
    name: str
    close: float
    entry: float
    ma5: float
    ma10: float
    ma20: float
    ma60: float
    ma200: float
    day: date


@dataclass
class Align200Trade:
    signal: Align200Signal
    exit_price: float
    exit_time: pd.Timestamp
    exit_reason: str
    pnl_pct: float
    pnl_pct_net: float


def add_align_features(df: pd.DataFrame) -> pd.DataFrame:
    """一分K均線：MA200 = 近 200 根 1m 收盤平均，不是日線 200。"""
    out = df.copy()
    close = out["close"]
    for period in MA_PERIODS:
        out[f"ma{period}"] = close.rolling(period, min_periods=period).mean()
    out["session_date"] = [pd.Timestamp(ts).date() for ts in out.index]
    return out


def is_aligned(row: pd.Series) -> bool:
    vals = [row.get("ma5"), row.get("ma10"), row.get("ma20"), row.get("ma60"), row.get("ma200")]
    if any(pd.isna(v) for v in vals):
        return False
    return bool(row["ma5"] > row["ma10"] > row["ma20"] > row["ma60"] and row["close"] > row["ma200"])


def detect_align200(
    df: pd.DataFrame,
    *,
    symbol: str = "",
    name: str = "",
    on_date: date | None = None,
    entry_after_minute: int = 10,
    one_per_day: bool = True,
) -> list[Align200Signal]:
    """條件剛成立的那一根：5>10>20>60 且收盤站上 200。前一根尚未同時滿足。"""
    if len(df) < 201:
        return []
    work = add_align_features(df)
    signals: list[Align200Signal] = []
    seen_days: set[date] = set()

    for i in range(1, len(work)):
        ts = work.index[i]
        day = pd.Timestamp(ts).date()
        if on_date is not None and day != on_date:
            continue
        if ts.hour < 9 or (ts.hour == 9 and ts.minute < entry_after_minute):
            continue
        if ts.hour > 13 or (ts.hour == 13 and ts.minute >= 20):
            continue
        if one_per_day and day in seen_days:
            continue
        row = work.iloc[i]
        prev = work.iloc[i - 1]
        if not is_aligned(row):
            continue
        if is_aligned(prev) and pd.Timestamp(work.index[i - 1]).date() == day:
            continue
        if i + 1 >= len(work):
            entry = float(row["close"])
        else:
            entry = float(work["open"].iloc[i + 1])
        signals.append(
            Align200Signal(
                timestamp=ts,
                bar_idx=i,
                symbol=symbol,
                name=name,
                close=float(row["close"]),
                entry=entry,
                ma5=float(row["ma5"]),
                ma10=float(row["ma10"]),
                ma20=float(row["ma20"]),
                ma60=float(row["ma60"]),
                ma200=float(row["ma200"]),
                day=day,
            )
        )
        seen_days.add(day)
    return signals


def run_align200_backtest(
    df: pd.DataFrame,
    signals: list[Align200Signal],
    *,
    cost_bps: float = 8.0,
    flatten_hm: tuple[int, int] = (13, 20),
) -> list[Align200Trade]:
    if not signals:
        return []
    work = add_align_features(df)
    trades: list[Align200Trade] = []
    busy_until = -1
    for sig in signals:
        entry_idx = min(sig.bar_idx + 1, len(work) - 1)
        if entry_idx <= busy_until:
            continue
        entry = float(work["open"].iloc[entry_idx])
        exit_price = float(work["close"].iloc[-1])
        exit_time = work.index[-1]
        exit_reason = "time_stop"
        exit_idx = len(work) - 1
        for i in range(entry_idx, len(work)):
            ts = work.index[i]
            row = work.iloc[i]
            if (ts.hour, ts.minute) >= flatten_hm:
                exit_price = float(row["close"])
                exit_time = ts
                exit_reason = "session_flat"
                exit_idx = i
                break
            if i > entry_idx and (float(row["close"]) < float(row["ma200"]) or float(row["ma5"]) < float(row["ma10"])):
                exit_price = float(row["close"])
                exit_time = ts
                exit_reason = "lost_align"
                exit_idx = i
                break
        busy_until = exit_idx
        pnl = (exit_price - entry) / entry if entry else 0.0
        trades.append(
            Align200Trade(
                signal=Align200Signal(
                    timestamp=work.index[entry_idx],
                    bar_idx=entry_idx,
                    symbol=sig.symbol,
                    name=sig.name,
                    close=sig.close,
                    entry=entry,
                    ma5=sig.ma5,
                    ma10=sig.ma10,
                    ma20=sig.ma20,
                    ma60=sig.ma60,
                    ma200=sig.ma200,
                    day=sig.day,
                ),
                exit_price=exit_price,
                exit_time=exit_time,
                exit_reason=exit_reason,
                pnl_pct=pnl,
                pnl_pct_net=pnl - 2.0 * (cost_bps / 10_000.0),
            )
        )
    return trades


def summarize_align(trades: list[Align200Trade]) -> dict:
    if not trades:
        return {
            "trades": 0,
            "wins": 0,
            "win_rate": 0.0,
            "total_pnl_pct_net": 0.0,
            "expectancy_net": 0.0,
            "by_day": {},
            "by_exit": {},
        }
    wins = sum(1 for t in trades if t.pnl_pct_net > 0)
    total = sum(t.pnl_pct_net for t in trades)
    by_day: dict[date, int] = {}
    by_exit: dict[str, int] = {}
    for t in trades:
        by_day[t.signal.day] = by_day.get(t.signal.day, 0) + 1
        by_exit[t.exit_reason] = by_exit.get(t.exit_reason, 0) + 1
    return {
        "trades": len(trades),
        "wins": wins,
        "win_rate": wins / len(trades),
        "total_pnl_pct_net": total,
        "expectancy_net": total / len(trades),
        "by_day": by_day,
        "by_exit": by_exit,
    }


def format_alert(sig: Align200Signal) -> str:
    return (
        f"{sig.symbol} {sig.name}  1m 站上MA200\n"
        f"收 {sig.close:.2f}  進場參考 {sig.entry:.2f}\n"
        f"MA5 {sig.ma5:.2f} > MA10 {sig.ma10:.2f} > MA20 {sig.ma20:.2f} > MA60 {sig.ma60:.2f}\n"
        f"MA200 {sig.ma200:.2f}　{sig.timestamp}"
    )

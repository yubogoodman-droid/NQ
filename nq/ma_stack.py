"""NQ 一分 K：均線多頭排列（短均在上、長均在下打開）。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MA_PERIODS = (5, 10, 20, 30, 60, 100, 120, 200)
SHORT = (5, 10, 20)
MID = (5, 10, 20, 30, 60)
FULL = MA_PERIODS


@dataclass(frozen=True)
class StackSignal:
    idx: int
    timestamp: pd.Timestamp
    entry: float
    fan_pct: float
    order: tuple[int, ...]
    level: str  # short / mid / full

    @property
    def order_text(self) -> str:
        return ">".join(str(n) for n in self.order)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for n in MA_PERIODS:
        out[f"ma{n}"] = out["close"].rolling(n, min_periods=n).mean()
    out["fan_pct"] = (out["ma5"] / out["ma200"] - 1.0) * 100.0
    out["stack_short"] = _stacked(out, SHORT)
    out["stack_mid"] = _stacked(out, MID)
    out["stack_full"] = _stacked(out, FULL)
    above = pd.Series(True, index=out.index)
    for n in MA_PERIODS:
        above &= out["close"] > out[f"ma{n}"]
    out["above_all"] = above
    return out


def _stacked(df: pd.DataFrame, periods: tuple[int, ...]) -> pd.Series:
    ok = pd.Series(True, index=df.index)
    for a, b in zip(periods, periods[1:]):
        ok &= df[f"ma{a}"] > df[f"ma{b}"]
    return ok


def _rising(mask: pd.Series) -> pd.Series:
    prev = mask.shift(1)
    return mask & prev.eq(False)


def _cooldown(idxs: np.ndarray, gap: int) -> list[int]:
    kept: list[int] = []
    last = -10_000
    for i in idxs:
        i = int(i)
        if i - last >= gap:
            kept.append(i)
            last = i
    return kept


def ma_order(row: pd.Series) -> tuple[int, ...]:
    pairs = []
    for n in MA_PERIODS:
        val = row.get(f"ma{n}")
        if pd.isna(val):
            return ()
        pairs.append((n, float(val)))
    pairs.sort(key=lambda x: -x[1])
    return tuple(n for n, _ in pairs)


def count_stack_events(
    df: pd.DataFrame,
    *,
    level: str = "full",
    cooldown: int = 30,
    require_price_above: bool = True,
) -> list[StackSignal]:
    """第一次形成指定均線排列的 K（預設 30 分鐘內不重複）。"""
    if "stack_full" not in df.columns:
        df = add_indicators(df)
    col = {"short": "stack_short", "mid": "stack_mid", "full": "stack_full"}[level]
    mask = df[col]
    if require_price_above:
        mask = mask & df["above_all"]
    first = _rising(mask)
    idxs = _cooldown(np.flatnonzero(first.to_numpy()), cooldown)
    signals: list[StackSignal] = []
    for i in idxs:
        row = df.iloc[i]
        fan = row.get("fan_pct")
        signals.append(
            StackSignal(
                idx=i,
                timestamp=df.index[i],
                entry=float(row["close"]),
                fan_pct=float(fan) if pd.notna(fan) else 0.0,
                order=ma_order(row),
                level=level,
            )
        )
    return signals


def ladder_counts(df: pd.DataFrame, *, cooldown: int = 30) -> dict[str, int]:
    return {
        "short": len(count_stack_events(df, level="short", cooldown=cooldown)),
        "mid": len(count_stack_events(df, level="mid", cooldown=cooldown)),
        "full": len(count_stack_events(df, level="full", cooldown=cooldown)),
    }

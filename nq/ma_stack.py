"""NQ 一分 K：急跌之後的均線多頭排列。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MA_PERIODS = (5, 10, 20, 30, 60, 100, 120, 200)
SHORT = (5, 10, 20)
MID = (5, 10, 20, 30, 60)
FULL = MA_PERIODS
TICK_SIZE = 0.25


@dataclass(frozen=True)
class DumpSpec:
    min_range_atr: float = 5.0
    min_range_pts: float = 50.0
    min_vol_ratio: float = 5.0
    require_below_all: bool = True
    cluster_gap: int = 30


STRICT_DUMP = DumpSpec()
MID_DUMP = DumpSpec(min_range_atr=3.0, min_range_pts=25.0, min_vol_ratio=3.0)
LOOSE_DUMP = DumpSpec(
    min_range_atr=2.5,
    min_range_pts=0.0,
    min_vol_ratio=2.5,
    require_below_all=False,
)


@dataclass(frozen=True)
class DumpEvent:
    idx: int
    timestamp: pd.Timestamp
    low: float
    high: float
    close: float
    range_pts: float
    vol_ratio: float
    range_atr: float


@dataclass(frozen=True)
class StackSignal:
    idx: int
    timestamp: pd.Timestamp
    entry: float
    fan_pct: float
    order: tuple[int, ...]
    level: str

    @property
    def order_text(self) -> str:
        return ">".join(str(n) for n in self.order)


@dataclass
class ComboSignal:
    dump: DumpEvent
    broke_low: bool
    short: StackSignal | None = None
    mid: StackSignal | None = None
    full: StackSignal | None = None

    @property
    def aligned(self) -> bool:
        return self.short is not None and not self.broke_low


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for n in MA_PERIODS:
        out[f"ma{n}"] = out["close"].rolling(n, min_periods=n).mean()
    out["fan_pct"] = (out["ma5"] / out["ma200"] - 1.0) * 100.0
    out["stack_short"] = _stacked(out, SHORT)
    out["stack_mid"] = _stacked(out, MID)
    out["stack_full"] = _stacked(out, FULL)
    above = pd.Series(True, index=out.index)
    below = pd.Series(True, index=out.index)
    for n in MA_PERIODS:
        above &= out["close"] > out[f"ma{n}"]
        below &= out["close"] < out[f"ma{n}"]
    out["above_all"] = above
    out["below_all"] = below
    out["range"] = out["high"] - out["low"]
    out["v20"] = out["volume"].rolling(20, min_periods=20).mean()
    out["atr20"] = out["range"].rolling(20, min_periods=20).mean()
    out["vol_ratio"] = out["volume"] / out["v20"]
    out["range_atr"] = out["range"] / out["atr20"]
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


def _stack_at(df: pd.DataFrame, i: int, level: str) -> StackSignal:
    row = df.iloc[i]
    fan = row.get("fan_pct")
    return StackSignal(
        idx=i,
        timestamp=df.index[i],
        entry=float(row["close"]),
        fan_pct=float(fan) if pd.notna(fan) else 0.0,
        order=ma_order(row),
        level=level,
    )


def count_stack_events(
    df: pd.DataFrame,
    *,
    level: str = "full",
    cooldown: int = 30,
    require_price_above: bool = True,
) -> list[StackSignal]:
    if "stack_full" not in df.columns:
        df = add_indicators(df)
    col = {"short": "stack_short", "mid": "stack_mid", "full": "stack_full"}[level]
    mask = df[col]
    if require_price_above:
        mask = mask & df["above_all"]
    first = _rising(mask)
    idxs = _cooldown(np.flatnonzero(first.to_numpy()), cooldown)
    return [_stack_at(df, i, level) for i in idxs]


def ladder_counts(df: pd.DataFrame, *, cooldown: int = 30) -> dict[str, int]:
    return {
        "short": len(count_stack_events(df, level="short", cooldown=cooldown)),
        "mid": len(count_stack_events(df, level="mid", cooldown=cooldown)),
        "full": len(count_stack_events(df, level="full", cooldown=cooldown)),
    }


def find_dumps(df: pd.DataFrame, spec: DumpSpec = STRICT_DUMP) -> list[DumpEvent]:
    if "vol_ratio" not in df.columns:
        df = add_indicators(df)
    red = df["close"] < df["open"]
    vol_ok = df["vol_ratio"] >= spec.min_vol_ratio
    range_ok = df["range_atr"] >= spec.min_range_atr
    if spec.min_range_pts > 0:
        range_ok = range_ok | (df["range"] >= spec.min_range_pts)
    loc = df["below_all"] if spec.require_below_all else df["close"] < df["ma20"]
    mask = (red & vol_ok & range_ok & loc).fillna(False)
    events: list[DumpEvent] = []
    for i in _cooldown(np.flatnonzero(mask.to_numpy()), spec.cluster_gap):
        row = df.iloc[i]
        events.append(
            DumpEvent(
                idx=i,
                timestamp=df.index[i],
                low=float(row["low"]),
                high=float(row["high"]),
                close=float(row["close"]),
                range_pts=float(row["range"]),
                vol_ratio=float(row["vol_ratio"]),
                range_atr=float(row["range_atr"]),
            )
        )
    return events


def scan_dump(
    df: pd.DataFrame,
    dump: DumpEvent,
    *,
    max_bars: int = 90,
) -> ComboSignal:
    n = len(df)
    end = min(n - 1, dump.idx + max_bars)
    combo = ComboSignal(dump=dump, broke_low=False)
    for j in range(dump.idx + 1, end + 1):
        gap = (df.index[j] - df.index[j - 1]).total_seconds()
        if gap > 5 * 60:
            break
        row = df.iloc[j]
        if row["low"] < dump.low - TICK_SIZE:
            combo.broke_low = True
            break
        above = bool(row["above_all"])
        if combo.short is None and above and bool(row["stack_short"]):
            combo.short = _stack_at(df, j, "short")
        if combo.mid is None and above and bool(row["stack_mid"]):
            combo.mid = _stack_at(df, j, "mid")
        if combo.full is None and above and bool(row["stack_full"]):
            combo.full = _stack_at(df, j, "full")
    return combo


def dump_align_ladder(
    df: pd.DataFrame,
    spec: DumpSpec = STRICT_DUMP,
    *,
    max_bars: int = 90,
) -> dict:
    dumps = find_dumps(df, spec)
    combos = [scan_dump(df, d, max_bars=max_bars) for d in dumps]
    v = [c for c in combos if not c.broke_low]
    return {
        "dumps": len(dumps),
        "v": len(v),
        "short": sum(1 for c in v if c.short),
        "mid": sum(1 for c in v if c.mid),
        "full": sum(1 for c in v if c.full),
        "combos": combos,
        "signals": [c for c in v if c.short],
    }

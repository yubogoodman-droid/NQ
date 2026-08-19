"""NQ 一分 K：急跌 V 反 → 站回均線 → 短均開花。"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

MA_PERIODS = (5, 10, 20, 30, 60, 100, 120, 200)
TICK_SIZE = 0.25


@dataclass(frozen=True)
class DumpSpec:
    """急跌 K 的門檻。range 條件為 ATR 倍數或絕對點數擇一即可。"""

    min_range_atr: float = 5.0
    min_range_pts: float = 50.0
    min_vol_ratio: float = 5.0
    require_below_all_ma: bool = True
    require_red: bool = True
    cluster_gap: int = 30


STRICT = DumpSpec()
MID = DumpSpec(min_range_atr=3.0, min_range_pts=25.0, min_vol_ratio=3.0, require_below_all_ma=True)
LOOSE = DumpSpec(
    min_range_atr=2.5,
    min_range_pts=0.0,
    min_vol_ratio=2.5,
    require_below_all_ma=False,
)


@dataclass(frozen=True)
class DumpEvent:
    idx: int
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    range_pts: float
    vol_ratio: float
    range_atr: float
    below_all_ma: bool


@dataclass(frozen=True)
class VRecoverySignal:
    dump: DumpEvent
    entry_idx: int
    entry_time: pd.Timestamp
    entry: float
    stop_loss: float
    prev_close: float | None
    recover_frac: float
    prev_close_idx: int | None = None

    @property
    def risk(self) -> float:
        return self.entry - self.stop_loss


@dataclass
class LadderCounts:
    dumps: int = 0
    v70: int = 0
    reclaim: int = 0
    reclaim_fan: int = 0
    prev_close: int = 0
    signals: list[VRecoverySignal] = field(default_factory=list)
    dumps_list: list[DumpEvent] = field(default_factory=list)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for n in MA_PERIODS:
        out[f"ma{n}"] = out["close"].rolling(n, min_periods=n).mean()
    out["range"] = out["high"] - out["low"]
    out["v20"] = out["volume"].rolling(20, min_periods=20).mean()
    out["atr20"] = out["range"].rolling(20, min_periods=20).mean()
    out["vol_ratio"] = out["volume"] / out["v20"]
    out["range_atr"] = out["range"] / out["atr20"]
    out["prev_close"] = _session_prev_close(out)
    return out


def _session_prev_close(df: pd.DataFrame) -> pd.Series:
    """每個 bar 對應「最近一次已完成的 16:00 ET 收盤」。"""
    if df.empty:
        return pd.Series(dtype=float)
    idx = df.index
    mask = (idx.hour == 16) & (idx.minute == 0)
    closes = df.loc[mask, "close"]
    return closes.reindex(idx).ffill().shift(1)


def _row_below_all(row: pd.Series) -> bool:
    for n in MA_PERIODS:
        ma = row.get(f"ma{n}")
        if pd.isna(ma) or row["close"] >= ma:
            return False
    return True


def _row_above_all(row: pd.Series) -> bool:
    for n in MA_PERIODS:
        ma = row.get(f"ma{n}")
        if pd.isna(ma) or row["close"] <= ma:
            return False
    return True


def _fan_short(row: pd.Series) -> bool:
    m5, m10, m20 = row.get("ma5"), row.get("ma10"), row.get("ma20")
    if pd.isna(m5) or pd.isna(m10) or pd.isna(m20):
        return False
    return m5 > m10 > m20


def _is_dump_bar(row: pd.Series, spec: DumpSpec) -> bool:
    if pd.isna(row.get("vol_ratio")) or pd.isna(row.get("range_atr")) or pd.isna(row.get("ma20")):
        return False
    if spec.require_red and row["close"] >= row["open"]:
        return False
    if row["vol_ratio"] < spec.min_vol_ratio:
        return False
    range_ok = row["range_atr"] >= spec.min_range_atr
    if spec.min_range_pts > 0:
        range_ok = range_ok or row["range"] >= spec.min_range_pts
    if not range_ok:
        return False
    if spec.require_below_all_ma:
        return _row_below_all(row)
    return row["close"] < row["ma20"]


def find_dumps(df: pd.DataFrame, spec: DumpSpec = STRICT) -> list[DumpEvent]:
    events: list[DumpEvent] = []
    last = -10_000
    for i in range(len(df)):
        row = df.iloc[i]
        if not _is_dump_bar(row, spec):
            continue
        if i - last < spec.cluster_gap:
            continue
        events.append(
            DumpEvent(
                idx=i,
                timestamp=df.index[i],
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                range_pts=float(row["range"]),
                vol_ratio=float(row["vol_ratio"]),
                range_atr=float(row["range_atr"]),
                below_all_ma=_row_below_all(row),
            )
        )
        last = i
    return events


def _scan_recovery(
    df: pd.DataFrame,
    dump: DumpEvent,
    *,
    max_bars: int = 90,
    min_recover_frac: float = 0.70,
) -> dict:
    n = len(df)
    dump_low = dump.low
    dump_range = dump.range_pts
    end = min(n - 1, dump.idx + max_bars)
    out = {
        "broke_low": False,
        "v70_idx": None,
        "reclaim_idx": None,
        "fan_idx": None,
        "signal_idx": None,
        "prev_close_idx": None,
        "recover_frac": 0.0,
    }
    if dump_range <= 0:
        return out

    peak_frac = 0.0
    for j in range(dump.idx + 1, end + 1):
        row = df.iloc[j]
        if row["low"] < dump_low - TICK_SIZE:
            out["broke_low"] = True
            break
        frac = (row["close"] - dump_low) / dump_range
        if frac > peak_frac:
            peak_frac = frac
        if out["v70_idx"] is None and frac >= min_recover_frac:
            out["v70_idx"] = j
        above = _row_above_all(row)
        fan = _fan_short(row)
        if out["reclaim_idx"] is None and above:
            out["reclaim_idx"] = j
        if out["fan_idx"] is None and fan:
            out["fan_idx"] = j
        if out["signal_idx"] is None and above and fan:
            out["signal_idx"] = j
        pc = row.get("prev_close")
        if (
            out["prev_close_idx"] is None
            and out["signal_idx"] is not None
            and pd.notna(pc)
            and row["close"] > pc
        ):
            out["prev_close_idx"] = j
    out["recover_frac"] = peak_frac
    return out


def count_ladder(
    df: pd.DataFrame,
    spec: DumpSpec = STRICT,
    *,
    max_bars: int = 90,
    min_recover_frac: float = 0.70,
) -> LadderCounts:
    if "ma200" not in df.columns:
        df = add_indicators(df)
    dumps = find_dumps(df, spec)
    ladder = LadderCounts(dumps=len(dumps), dumps_list=dumps)
    for dump in dumps:
        rec = _scan_recovery(df, dump, max_bars=max_bars, min_recover_frac=min_recover_frac)
        if rec["broke_low"]:
            continue
        if rec["v70_idx"] is not None:
            ladder.v70 += 1
        if rec["reclaim_idx"] is not None:
            ladder.reclaim += 1
        if rec["signal_idx"] is None:
            continue
        ladder.reclaim_fan += 1
        if rec["prev_close_idx"] is not None:
            ladder.prev_close += 1
        entry_idx = rec["signal_idx"]
        entry_row = df.iloc[entry_idx]
        pc = entry_row.get("prev_close")
        ladder.signals.append(
            VRecoverySignal(
                dump=dump,
                entry_idx=entry_idx,
                entry_time=df.index[entry_idx],
                entry=float(entry_row["close"]),
                stop_loss=dump.low,
                prev_close=None if pd.isna(pc) else float(pc),
                recover_frac=float(rec["recover_frac"]),
                prev_close_idx=rec["prev_close_idx"],
            )
        )
    return ladder

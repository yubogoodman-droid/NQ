"""1 分 K：MA5/10/20 多頭排列，且收盤站上 MA200 時發出通知。"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class MaAlignPattern:
    """多頭排列下，收盤剛站上 MA200。"""

    bar_idx: int
    close: float
    ma5: float
    ma10: float
    ma20: float
    ma200: float

    @property
    def stop_loss(self) -> float:
        """停損放在 MA20。"""
        return self.ma20


def add_mas(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ma5"] = out["close"].rolling(5, min_periods=5).mean()
    out["ma10"] = out["close"].rolling(10, min_periods=10).mean()
    out["ma20"] = out["close"].rolling(20, min_periods=20).mean()
    out["ma200"] = out["close"].rolling(200, min_periods=200).mean()
    return out


add_daily_mas = add_mas


def _bull_stack(ma5: float, ma10: float, ma20: float) -> bool:
    return ma5 > ma10 > ma20


def _as_local(ts: pd.Timestamp) -> pd.Timestamp:
    if getattr(ts, "tzinfo", None) is not None:
        return ts.tz_convert("Asia/Taipei")
    return ts


def is_stacked(row: pd.Series) -> bool:
    if pd.isna(row.get("ma5")) or pd.isna(row.get("ma10")) or pd.isna(row.get("ma20")):
        return False
    return _bull_stack(float(row["ma5"]), float(row["ma10"]), float(row["ma20"]))


def is_above_ma200(row: pd.Series) -> bool:
    if pd.isna(row.get("ma200")):
        return False
    return float(row["close"]) > float(row["ma200"])


def is_aligned(row: pd.Series) -> bool:
    return is_stacked(row) and is_above_ma200(row)


def detect_ma_align_alerts(
    df: pd.DataFrame,
    *,
    skip_open_minutes: int = 5,
) -> list[MaAlignPattern]:
    """
    偵測收盤剛站上 1 分 K MA200 的通知：
    前一根收盤尚未站上 MA200，這一根收盤才站上，且當時 MA5 > MA10 > MA20。
    已經站在 MA200 上方、只是均線重新排好，不算。
    """
    if df.empty or "close" not in df.columns:
        return []
    work = add_mas(df)
    patterns: list[MaAlignPattern] = []
    prev_above = False
    for i in range(len(work)):
        row = work.iloc[i]
        above = is_above_ma200(row)
        stacked = is_stacked(row)
        too_early = False
        if skip_open_minutes > 0:
            local = _as_local(df.index[i])
            too_early = local.hour == 9 and local.minute < skip_open_minutes
        if stacked and above and not prev_above and not too_early:
            patterns.append(
                MaAlignPattern(
                    bar_idx=i,
                    close=float(row["close"]),
                    ma5=float(row["ma5"]),
                    ma10=float(row["ma10"]),
                    ma20=float(row["ma20"]),
                    ma200=float(row["ma200"]),
                )
            )
        prev_above = above
    return patterns

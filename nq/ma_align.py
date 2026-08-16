"""日線：MA5/10/20 多頭排列，且收盤站上 MA200 時發出通知。"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class MaAlignPattern:
    """剛成立的多頭排列 + 站上 200 日均。"""

    bar_idx: int
    close: float
    ma5: float
    ma10: float
    ma20: float
    ma200: float

    @property
    def stop_loss(self) -> float:
        """停損放在 20 日均下方（至少離進場 1.5%）。"""
        floor = self.close * 0.985
        return min(self.ma20, floor)


def add_daily_mas(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ma5"] = out["close"].rolling(5, min_periods=5).mean()
    out["ma10"] = out["close"].rolling(10, min_periods=10).mean()
    out["ma20"] = out["close"].rolling(20, min_periods=20).mean()
    out["ma200"] = out["close"].rolling(200, min_periods=200).mean()
    return out


def _bull_stack(ma5: float, ma10: float, ma20: float) -> bool:
    return ma5 > ma10 > ma20


def is_aligned(row: pd.Series) -> bool:
    if pd.isna(row.get("ma5")) or pd.isna(row.get("ma10")) or pd.isna(row.get("ma20")) or pd.isna(row.get("ma200")):
        return False
    return _bull_stack(float(row["ma5"]), float(row["ma10"]), float(row["ma20"])) and float(row["close"]) > float(
        row["ma200"]
    )


def detect_ma_align_alerts(df: pd.DataFrame) -> list[MaAlignPattern]:
    """
    偵測「今天才成立」的通知：MA5 > MA10 > MA20 且收盤 > MA200，
    且前一日尚未同時滿足。
    """
    if df.empty or "close" not in df.columns:
        return []
    work = add_daily_mas(df)
    patterns: list[MaAlignPattern] = []
    prev_ok = False
    for i in range(len(work)):
        row = work.iloc[i]
        ok = is_aligned(row)
        if ok and not prev_ok:
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
        prev_ok = ok
    return patterns
